"""Shared compiled problem-store helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..config.models import SplitConfig

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(slots=True)
class DatasetSplitIndices:
    train: IntVector
    validation: IntVector
    test: IntVector


@dataclass(frozen=True, slots=True)
class ContextWindowSummary:
    sample_indices: IntVector
    context_start_rows: IntVector
    anchor_rows: IntVector
    context_lengths: IntVector


@dataclass(frozen=True, slots=True)
class CandidateWindowSummary:
    sample_indices: IntVector
    anchor_rows: IntVector
    baseline_rows: IntVector
    candidate_end_rows: IntVector
    candidate_counts: IntVector
    reachable_end_rows: IntVector
    last_candidate_rows: IntVector


@dataclass(slots=True)
class CompiledProblemStore:
    feature_matrix: FloatMatrix
    log_base_fees: FloatVector
    timestamps: IntVector
    anchor_rows: IntVector
    context_start_rows: IntVector
    candidate_start_rows: IntVector
    candidate_end_rows: IntVector
    max_candidate_slots: int

    @property
    def n_rows(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.feature_matrix.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.anchor_rows.shape[0])

    def tail_sample_indices(self, *, sample_count: int) -> IntVector:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.n_samples < sample_count:
            raise ValueError(
                "History dataset is too short for the requested sample count; "
                f"need at least {sample_count} valid anchors, got {self.n_samples}"
            )
        return np.arange(self.n_samples - sample_count, self.n_samples, dtype=np.int64)

    def with_fixed_context_length(
        self,
        *,
        context_length: int,
        history_seconds: int,
        warmup_rows: int,
    ) -> CompiledProblemStore:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        context_start_rows = self.anchor_rows - context_length + 1
        valid_anchor_mask = context_start_rows >= 0
        valid_anchor_mask &= context_start_rows >= warmup_rows
        if history_seconds > 0:
            valid_anchor_mask &= (
                self.timestamps[np.maximum(context_start_rows, 0)] - self.timestamps[0]
            ) >= history_seconds
        anchor_rows = self.anchor_rows[valid_anchor_mask].astype(np.int64, copy=False)
        if anchor_rows.size == 0:
            raise ValueError("fixed context length produced no supervised samples")
        return replace(
            self,
            anchor_rows=anchor_rows,
            context_start_rows=context_start_rows[valid_anchor_mask].astype(
                np.int64,
                copy=False,
            ),
            candidate_start_rows=self.candidate_start_rows[valid_anchor_mask].astype(
                np.int64,
                copy=False,
            ),
            candidate_end_rows=self.candidate_end_rows[valid_anchor_mask].astype(
                np.int64,
                copy=False,
            ),
        )

    def sample_indices_by_timestamp_window(
        self,
        *,
        start_timestamp: int,
        end_timestamp: int,
    ) -> IntVector:
        sample_timestamps = self.sample_timestamps(np.arange(self.n_samples, dtype=np.int64))
        mask = (sample_timestamps >= start_timestamp) & (sample_timestamps < end_timestamp)
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    def sample_timestamps(self, sample_indices: IntVector) -> IntVector:
        resolved_indices = sample_indices.astype(np.int64, copy=False)
        return self.timestamps[self.anchor_rows[resolved_indices]].astype(
            np.int64,
            copy=False,
        )

    def context_windows(self, sample_indices: IntVector) -> ContextWindowSummary:
        resolved_indices = sample_indices.astype(np.int64, copy=False)
        context_start_rows = self.context_start_rows[resolved_indices].astype(
            np.int64,
            copy=False,
        )
        anchor_rows = self.anchor_rows[resolved_indices].astype(np.int64, copy=False)
        return ContextWindowSummary(
            sample_indices=resolved_indices,
            context_start_rows=context_start_rows,
            anchor_rows=anchor_rows,
            context_lengths=(anchor_rows - context_start_rows + 1).astype(
                np.int64,
                copy=False,
            ),
        )

    def candidate_windows(self, sample_indices: IntVector) -> CandidateWindowSummary:
        resolved_indices = sample_indices.astype(np.int64, copy=False)
        if self.max_candidate_slots <= 0:
            raise ValueError("max_candidate_slots must be positive")
        anchor_rows = self.anchor_rows[resolved_indices].astype(np.int64, copy=False)
        baseline_rows = self.candidate_start_rows[resolved_indices].astype(
            np.int64,
            copy=False,
        )
        candidate_end_rows = self.candidate_end_rows[resolved_indices].astype(
            np.int64,
            copy=False,
        )
        candidate_counts = (candidate_end_rows - baseline_rows).astype(
            np.int64,
            copy=False,
        )
        if np.any(candidate_counts <= 0):
            raise ValueError("candidate windows require at least one candidate row")
        reachable_end_rows = np.minimum(
            candidate_end_rows,
            baseline_rows + int(self.max_candidate_slots),
        ).astype(np.int64, copy=False)
        last_candidate_rows = (reachable_end_rows - 1).astype(np.int64, copy=False)
        return CandidateWindowSummary(
            sample_indices=resolved_indices,
            anchor_rows=anchor_rows,
            baseline_rows=baseline_rows,
            candidate_end_rows=candidate_end_rows,
            candidate_counts=candidate_counts,
            reachable_end_rows=reachable_end_rows,
            last_candidate_rows=last_candidate_rows,
        )

    def selected_row_span(self, sample_indices: IntVector) -> tuple[int, int]:
        if sample_indices.size == 0:
            raise ValueError("sample_indices must be non-empty")
        resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
        first_sample = int(resolved_sample_indices[0])
        last_sample = int(resolved_sample_indices[-1])
        start = int(self.context_start_rows[first_sample])
        end = int(self.candidate_end_rows[last_sample])
        return start, end


def chronological_split_indices(
    n_samples: int,
    split_config: SplitConfig,
) -> DatasetSplitIndices:
    if n_samples < 3:
        raise ValueError("Need at least three examples to create train/validation/test splits")

    train_end = int(n_samples * split_config.train_fraction)
    validation_end = train_end + int(n_samples * split_config.validation_fraction)
    train_end = max(1, min(train_end, n_samples - 2))
    validation_end = max(train_end + 1, min(validation_end, n_samples - 1))
    all_indices = np.arange(n_samples, dtype=np.int64)
    return DatasetSplitIndices(
        train=all_indices[:train_end],
        validation=all_indices[train_end:validation_end],
        test=all_indices[validation_end:],
    )

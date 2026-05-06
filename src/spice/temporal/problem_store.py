"""Shared compiled problem-store helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


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
    candidate_start_rows: IntVector
    candidate_end_rows: IntVector
    candidate_counts: IntVector
    reachable_end_rows: IntVector


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

    def __post_init__(self) -> None:
        feature_matrix = np.asarray(self.feature_matrix, dtype=np.float32)
        log_base_fees = np.asarray(self.log_base_fees, dtype=np.float32)
        timestamps = np.asarray(self.timestamps, dtype=np.int64)
        anchor_rows = np.asarray(self.anchor_rows, dtype=np.int64)
        context_start_rows = np.asarray(self.context_start_rows, dtype=np.int64)
        candidate_start_rows = np.asarray(self.candidate_start_rows, dtype=np.int64)
        candidate_end_rows = np.asarray(self.candidate_end_rows, dtype=np.int64)
        max_candidate_slots = int(self.max_candidate_slots)

        if feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be a two-dimensional array")
        if feature_matrix.shape[0] == 0:
            raise ValueError("feature_matrix must contain at least one row")
        if feature_matrix.shape[1] == 0:
            raise ValueError("feature_matrix must contain at least one feature")
        if not np.all(np.isfinite(feature_matrix)):
            raise ValueError("feature_matrix must contain only finite values")

        n_rows = int(feature_matrix.shape[0])
        _require_1d_row_aligned(log_base_fees, n_rows, name="log_base_fees")
        if not np.all(np.isfinite(log_base_fees)):
            raise ValueError("log_base_fees must contain only finite values")
        _require_1d_row_aligned(timestamps, n_rows, name="timestamps")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be sorted in nondecreasing order")

        sample_count = _require_aligned_sample_rows(
            anchor_rows=anchor_rows,
            context_start_rows=context_start_rows,
            candidate_start_rows=candidate_start_rows,
            candidate_end_rows=candidate_end_rows,
        )
        if sample_count == 0:
            raise ValueError("CompiledProblemStore must contain at least one sample")
        if max_candidate_slots <= 0:
            raise ValueError("max_candidate_slots must be positive")

        if np.any(context_start_rows < 0):
            raise ValueError("context_start_rows must be non-negative")
        if np.any(anchor_rows < 0) or np.any(anchor_rows >= n_rows):
            raise ValueError("anchor_rows must be within store rows")
        if np.any(context_start_rows > anchor_rows):
            raise ValueError("context_start_rows must be <= anchor_rows")
        if np.any(candidate_start_rows < anchor_rows):
            raise ValueError("candidate_start_rows must be >= anchor_rows")
        if np.any(candidate_start_rows >= candidate_end_rows):
            raise ValueError("candidate_start_rows must be < candidate_end_rows")
        if np.any(candidate_end_rows > n_rows):
            raise ValueError("candidate_end_rows must be <= store row count")

        self.feature_matrix = feature_matrix
        self.log_base_fees = log_base_fees
        self.timestamps = timestamps
        self.anchor_rows = anchor_rows
        self.context_start_rows = context_start_rows
        self.candidate_start_rows = candidate_start_rows
        self.candidate_end_rows = candidate_end_rows
        self.max_candidate_slots = max_candidate_slots

    @property
    def n_rows(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.feature_matrix.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.anchor_rows.shape[0])

    def sample_timestamps(self, sample_indices: IntVector) -> IntVector:
        resolved_indices = self._validated_sample_indices(sample_indices)
        return self.timestamps[self.anchor_rows[resolved_indices]].astype(
            np.int64,
            copy=False,
        )

    def context_windows(self, sample_indices: IntVector) -> ContextWindowSummary:
        resolved_indices = self._validated_sample_indices(sample_indices)
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
        resolved_indices = self._validated_sample_indices(sample_indices)
        anchor_rows = self.anchor_rows[resolved_indices].astype(np.int64, copy=False)
        candidate_start_rows = self.candidate_start_rows[resolved_indices].astype(
            np.int64,
            copy=False,
        )
        candidate_end_rows = self.candidate_end_rows[resolved_indices].astype(
            np.int64,
            copy=False,
        )
        candidate_counts = (candidate_end_rows - candidate_start_rows).astype(
            np.int64,
            copy=False,
        )
        reachable_end_rows = np.minimum(
            candidate_end_rows,
            candidate_start_rows + int(self.max_candidate_slots),
        ).astype(np.int64, copy=False)
        return CandidateWindowSummary(
            sample_indices=resolved_indices,
            anchor_rows=anchor_rows,
            candidate_start_rows=candidate_start_rows,
            candidate_end_rows=candidate_end_rows,
            candidate_counts=candidate_counts,
            reachable_end_rows=reachable_end_rows,
        )

    def selected_row_span(self, sample_indices: IntVector) -> tuple[int, int]:
        resolved_sample_indices = self._validated_sample_indices(
            sample_indices,
            require_nonempty=True,
        )
        start = int(self.context_start_rows[resolved_sample_indices].min())
        end = int(self.candidate_end_rows[resolved_sample_indices].max())
        return start, end

    def context_row_multiplicities(self, sample_indices: IntVector) -> IntVector:
        resolved_indices = self._validated_sample_indices(
            sample_indices,
            require_nonempty=True,
        )
        counts = np.zeros(self.n_rows + 1, dtype=np.int64)
        starts = self.context_start_rows[resolved_indices]
        ends = self.anchor_rows[resolved_indices] + 1
        np.add.at(counts, starts, 1)
        np.add.at(counts, ends, -1)
        return np.cumsum(counts[:-1], dtype=np.int64)

    def _validated_sample_indices(
        self,
        sample_indices: IntVector,
        *,
        require_nonempty: bool = False,
    ) -> IntVector:
        resolved_indices = np.asarray(sample_indices, dtype=np.int64)
        if resolved_indices.ndim != 1:
            raise ValueError("sample_indices must be a one-dimensional array")
        if require_nonempty and resolved_indices.size == 0:
            raise ValueError("sample_indices must be non-empty")
        if np.any(resolved_indices < 0):
            raise ValueError("sample_indices must be non-negative")
        if np.any(resolved_indices >= self.n_samples):
            raise ValueError("sample_indices must be within store samples")
        return resolved_indices


def _require_1d_row_aligned(array: np.ndarray, n_rows: int, *, name: str) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.shape[0] != n_rows:
        raise ValueError(f"{name} must align with feature_matrix rows")


def _require_aligned_sample_rows(
    *,
    anchor_rows: np.ndarray,
    context_start_rows: np.ndarray,
    candidate_start_rows: np.ndarray,
    candidate_end_rows: np.ndarray,
) -> int:
    sample_arrays = {
        "anchor_rows": anchor_rows,
        "context_start_rows": context_start_rows,
        "candidate_start_rows": candidate_start_rows,
        "candidate_end_rows": candidate_end_rows,
    }
    sample_count: int | None = None
    for name, array in sample_arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif array.shape[0] != sample_count:
            raise ValueError("sample row arrays must have identical length")
    assert sample_count is not None
    return sample_count

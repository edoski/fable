"""Timestamp-native problem compiler."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from ...features import (
    CompiledFeatureContract,
    FeaturePrerequisites,
    ResolvedFeatureTable,
)
from ..contracts import CompiledProblemContract, ProblemRuntimeMetadata, TimestampRuntimeMetadata
from ..problem_store import CompiledProblemStore
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config import ProblemSpec
    from ...config.models import ChainRuntimeSpec


@dataclass(frozen=True, slots=True)
class TimestampSamplingWindow:
    lookback_seconds: int
    delay_seconds: int
    feature_prerequisites: FeaturePrerequisites

    @property
    def required_history_seconds(self) -> int:
        return self.lookback_seconds + self.feature_prerequisites.history_seconds


class TimestampNativeCompilerConfig(ProblemCompilerConfig):
    id: str = "timestamp_native"


@dataclass(frozen=True, slots=True)
class TimestampNativeCompiledProblemContract(CompiledProblemContract):
    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
            return minimum_window
        warmup_seconds = ceil(self.warmup_rows * recent_block_interval_seconds)
        return max(
            minimum_window,
            self.required_history_seconds
            + self.max_delay_seconds
            + warmup_seconds
            + ceil(self.sample_count * recent_block_interval_seconds),
        )

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        store, _ = self.build_capability_store(feature_table)
        return store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> tuple[CompiledProblemStore, TimestampRuntimeMetadata]:
        return (
            _build_timestamp_problem_store(
                feature_table,
                window=TimestampSamplingWindow(
                    lookback_seconds=self.lookback_seconds,
                    delay_seconds=self.max_delay_seconds,
                    feature_prerequisites=self.feature_prerequisites,
                ),
            ),
            TimestampRuntimeMetadata(),
        )

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        if not isinstance(compiler_runtime_metadata, TimestampRuntimeMetadata):
            raise TypeError("timestamp_native requires TimestampRuntimeMetadata")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if delay_seconds > self.max_delay_seconds:
            raise ValueError(
                "delay_seconds exceeds problem capability: "
                f"{delay_seconds} > {self.max_delay_seconds}"
            )
        return _build_timestamp_problem_store(
            feature_table,
            window=TimestampSamplingWindow(
                lookback_seconds=self.lookback_seconds,
                delay_seconds=delay_seconds,
                feature_prerequisites=self.feature_prerequisites,
            ),
            max_candidate_slots=max_candidate_slots,
        )


def compile_problem(
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
    del chain_runtime
    return TimestampNativeCompiledProblemContract(
        compiler_id="timestamp_native",
        problem_id=problem.id,
        feature_set_id=feature_contract.feature_set_id,
        feature_family_id=feature_contract.feature_family_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_delay_seconds=problem.max_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
    )


def _build_timestamp_problem_store(
    feature_table: ResolvedFeatureTable,
    *,
    window: TimestampSamplingWindow,
    max_candidate_slots: int | None = None,
) -> CompiledProblemStore:
    if window.lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if window.delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")

    timestamps = feature_table.series.timestamps
    log_base_fees = feature_table.series.log_base_fees
    feature_matrix = feature_table.feature_matrix
    if timestamps.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    context_start_rows = np.searchsorted(
        timestamps,
        timestamps - window.lookback_seconds,
        side="left",
    ).astype(np.int64, copy=False)
    candidate_end_rows = np.searchsorted(
        timestamps,
        timestamps + window.delay_seconds,
        side="right",
    ).astype(np.int64, copy=False)
    anchor_candidates = np.arange(timestamps.shape[0], dtype=np.int64)
    candidate_counts = candidate_end_rows - (anchor_candidates + 1)
    context_history_ready = (
        timestamps[context_start_rows] - timestamps[0]
    ) >= window.feature_prerequisites.history_seconds
    warmup_ready = context_start_rows >= window.feature_prerequisites.warmup_rows
    valid_anchor_mask = context_history_ready & warmup_ready & (candidate_counts > 0)
    anchor_rows = anchor_candidates[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    selected_context_starts = context_start_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_ends = candidate_end_rows[anchor_rows].astype(np.int64, copy=False)
    candidate_starts = anchor_rows + 1
    selected_candidate_counts = selected_candidate_ends - candidate_starts
    resolved_max_candidate_slots = (
        int(selected_candidate_counts.max())
        if max_candidate_slots is None
        else int(max_candidate_slots)
    )
    if resolved_max_candidate_slots <= 0:
        raise ValueError("max_candidate_slots must be positive")
    if np.any(selected_candidate_counts > resolved_max_candidate_slots):
        raise ValueError("Configured max_candidate_slots is too small for this dataset")

    return CompiledProblemStore(
        feature_matrix=feature_matrix,
        log_base_fees=log_base_fees,
        timestamps=timestamps,
        anchor_rows=anchor_rows,
        context_start_rows=selected_context_starts,
        candidate_end_rows=selected_candidate_ends,
        max_candidate_slots=resolved_max_candidate_slots,
    )

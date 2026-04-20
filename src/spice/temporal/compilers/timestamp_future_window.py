"""Future-only timestamp-bounded problem compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from ...features import (
    CompiledFeatureContract,
    FeaturePrerequisites,
    ResolvedFeatureTable,
)
from ..contracts import (
    CompiledProblemContract,
    ProblemRuntimeMetadata,
    TimestampFutureWindowRuntimeMetadata,
)
from ..problem_store import CompiledProblemStore
from ..realization import CompiledRealizationPolicyContract
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config import ProblemSpec
    from ...config.models import ChainRuntimeSpec


class TimestampFutureWindowIntervalSource(StrEnum):
    CALIBRATED = "calibrated"
    NOMINAL_CHAIN_RUNTIME = "nominal_chain_runtime"


class TimestampFutureWindowCalibratedStatistic(StrEnum):
    MEDIAN = "median"
    MEAN = "mean"


class TimestampFutureWindowCompilerConfig(ProblemCompilerConfig):
    id: str = "timestamp_future_window"
    action_interval_source: TimestampFutureWindowIntervalSource = (
        TimestampFutureWindowIntervalSource.NOMINAL_CHAIN_RUNTIME
    )
    calibrated_interval_statistic: TimestampFutureWindowCalibratedStatistic = Field(
        default=TimestampFutureWindowCalibratedStatistic.MEDIAN
    )


@dataclass(frozen=True, slots=True)
class TimestampFutureWindowCompiledProblemContract(CompiledProblemContract):
    action_interval_source: TimestampFutureWindowIntervalSource
    calibrated_interval_statistic: TimestampFutureWindowCalibratedStatistic
    nominal_block_time_seconds: float | None

    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        bootstrap_interval_seconds = _resolve_bootstrap_interval_seconds(
            source=self.action_interval_source,
            recent_block_interval_seconds=recent_block_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
        )
        if bootstrap_interval_seconds is None:
            return minimum_window
        return max(
            minimum_window,
            self.required_history_seconds
            + self.max_delay_seconds
            + math.ceil((self.warmup_rows + self.sample_count + 1) * bootstrap_interval_seconds),
        )

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        store, _ = self.build_capability_store(feature_table)
        return store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> tuple[CompiledProblemStore, TimestampFutureWindowRuntimeMetadata]:
        calibrated_interval_seconds = _calibrate_observed_block_interval_seconds(
            feature_table,
            statistic=self.calibrated_interval_statistic,
        )
        action_interval_seconds = _resolve_runtime_interval_seconds(
            source=self.action_interval_source,
            calibrated_interval_seconds=calibrated_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
        )
        capability_action_count = _action_count_for_delay(
            self.max_delay_seconds,
            action_interval_seconds,
        )
        store = _build_timestamp_future_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=self.max_delay_seconds,
            fixed_action_count=capability_action_count,
        )
        return (
            store,
            TimestampFutureWindowRuntimeMetadata(
                calibrated_interval_seconds=calibrated_interval_seconds,
                action_interval_seconds=action_interval_seconds,
                capability_action_count=capability_action_count,
            ),
        )

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        if not isinstance(compiler_runtime_metadata, TimestampFutureWindowRuntimeMetadata):
            raise TypeError("timestamp_future_window requires TimestampFutureWindowRuntimeMetadata")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if delay_seconds > self.max_delay_seconds:
            raise ValueError(
                "delay_seconds exceeds problem capability: "
                f"{delay_seconds} > {self.max_delay_seconds}"
            )
        return _build_timestamp_future_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=delay_seconds,
            fixed_action_count=max_candidate_slots,
        )


def compile_problem(
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    realization_policy: CompiledRealizationPolicyContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
    compiler_config = TimestampFutureWindowCompilerConfig.model_validate(problem.compiler)
    nominal_block_time_seconds = (
        None if chain_runtime is None else float(chain_runtime.nominal_block_time_seconds)
    )
    if (
        compiler_config.action_interval_source
        is TimestampFutureWindowIntervalSource.NOMINAL_CHAIN_RUNTIME
        and (nominal_block_time_seconds is None or nominal_block_time_seconds <= 0)
    ):
        raise ValueError(
            "timestamp_future_window requires chain.runtime.nominal_block_time_seconds "
            "for nominal action-space resolution"
        )
    return TimestampFutureWindowCompiledProblemContract(
        compiler_id="timestamp_future_window",
        problem_id=problem.id,
        feature_set_id=feature_contract.feature_set_id,
        feature_family_id=feature_contract.feature_family_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_delay_seconds=problem.max_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
        realization_policy=realization_policy,
        action_interval_source=compiler_config.action_interval_source,
        calibrated_interval_statistic=compiler_config.calibrated_interval_statistic,
        nominal_block_time_seconds=nominal_block_time_seconds,
    )


def _build_timestamp_future_window_store(
    feature_table: ResolvedFeatureTable,
    *,
    feature_prerequisites: FeaturePrerequisites,
    lookback_seconds: int,
    delay_seconds: int,
    fixed_action_count: int,
) -> CompiledProblemStore:
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")
    if fixed_action_count <= 0:
        raise ValueError("fixed_action_count must be positive")
    timestamps = feature_table.series.timestamps
    if timestamps.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")
    context_start_rows = np.searchsorted(
        timestamps,
        timestamps - lookback_seconds,
        side="left",
    ).astype(np.int64, copy=False)
    candidate_end_rows = np.searchsorted(
        timestamps,
        timestamps + delay_seconds,
        side="right",
    ).astype(np.int64, copy=False)
    anchor_candidates = np.arange(timestamps.shape[0], dtype=np.int64)
    candidate_counts = candidate_end_rows - (anchor_candidates + 1)
    context_history_ready = (
        timestamps[context_start_rows] - timestamps[0]
    ) >= feature_prerequisites.history_seconds
    warmup_ready = context_start_rows >= feature_prerequisites.warmup_rows
    future_ready = candidate_counts > 0
    post_window_ready = candidate_end_rows < timestamps.shape[0]
    valid_anchor_mask = context_history_ready & warmup_ready & future_ready & post_window_ready
    anchor_rows = anchor_candidates[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")
    selected_context_starts = context_start_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_ends = candidate_end_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_counts = selected_candidate_ends - (anchor_rows + 1)
    if np.any(selected_candidate_counts > fixed_action_count):
        raise ValueError(
            "timestamp_future_window requires fixed action space to upper-bound realized "
            "future candidates"
        )
    return CompiledProblemStore(
        feature_matrix=feature_table.feature_matrix,
        log_base_fees=feature_table.series.log_base_fees,
        timestamps=timestamps,
        anchor_rows=anchor_rows,
        context_start_rows=selected_context_starts,
        candidate_end_rows=selected_candidate_ends,
        max_candidate_slots=fixed_action_count,
    )


def _calibrate_observed_block_interval_seconds(
    feature_table: ResolvedFeatureTable,
    *,
    statistic: TimestampFutureWindowCalibratedStatistic,
) -> float:
    deltas = np.diff(feature_table.series.timestamps.astype(np.int64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError("timestamp_future_window requires positive timestamp deltas")
    if statistic is TimestampFutureWindowCalibratedStatistic.MEAN:
        return float(np.mean(positive_deltas))
    return float(np.median(positive_deltas))


def _resolve_runtime_interval_seconds(
    *,
    source: TimestampFutureWindowIntervalSource,
    calibrated_interval_seconds: float,
    nominal_block_time_seconds: float | None,
) -> float:
    if source is TimestampFutureWindowIntervalSource.CALIBRATED:
        return calibrated_interval_seconds
    if nominal_block_time_seconds is None or nominal_block_time_seconds <= 0:
        raise ValueError(
            "timestamp_future_window requires nominal block time for action interval resolution"
        )
    return nominal_block_time_seconds


def _resolve_bootstrap_interval_seconds(
    *,
    source: TimestampFutureWindowIntervalSource,
    recent_block_interval_seconds: float | None,
    nominal_block_time_seconds: float | None,
) -> float | None:
    if source is TimestampFutureWindowIntervalSource.NOMINAL_CHAIN_RUNTIME:
        return nominal_block_time_seconds
    if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
        return None
    return recent_block_interval_seconds


def _action_count_for_delay(
    max_delay_seconds: int,
    interval_seconds: float,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return max(1, math.floor(max_delay_seconds / interval_seconds))

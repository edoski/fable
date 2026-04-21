"""Future-only timestamp-bounded problem compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from ...features import (
    CompiledFeatureContract,
    ResolvedFeatureTable,
)
from ..contracts import (
    CompiledProblemContract,
    ProblemRuntimeMetadata,
    TimestampFutureWindowRuntimeMetadata,
)
from ..problem_store import CompiledProblemStore
from ..realization import CompiledRealizationPolicyContract
from ._shared import (
    build_timestamp_window_store,
    calibrate_positive_timestamp_delta_seconds,
    resolve_bootstrap_interval_seconds,
    resolve_runtime_interval_seconds,
)
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec


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
        bootstrap_interval_seconds = resolve_bootstrap_interval_seconds(
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
        calibrated_interval_seconds = calibrate_positive_timestamp_delta_seconds(
            feature_table,
            statistic=self.calibrated_interval_statistic,
            empty_error="timestamp_future_window requires positive timestamp deltas",
        )
        action_interval_seconds = resolve_runtime_interval_seconds(
            source=self.action_interval_source,
            calibrated_interval_seconds=calibrated_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
            compiler_label="timestamp_future_window",
            interval_label="action",
        )
        capability_action_count = _action_count_for_delay(
            self.max_delay_seconds,
            action_interval_seconds,
        )
        store = build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=self.max_delay_seconds,
            fixed_candidate_count=capability_action_count,
            fixed_candidate_count_error=(
                "timestamp_future_window requires fixed action space to upper-bound realized "
                "future candidates"
            ),
            requires_post_window_row=self.realization_policy.requires_post_window_row,
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
        return build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=delay_seconds,
            fixed_candidate_count=max_candidate_slots,
            fixed_candidate_count_error=(
                "timestamp_future_window requires fixed action space to upper-bound realized "
                "future candidates"
            ),
            requires_post_window_row=self.realization_policy.requires_post_window_row,
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


def _action_count_for_delay(
    max_delay_seconds: int,
    interval_seconds: float,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return max(1, math.floor(max_delay_seconds / interval_seconds))

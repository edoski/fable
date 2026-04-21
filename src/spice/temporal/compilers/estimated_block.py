"""Corpus-calibrated estimated-block problem compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np
from pydantic import Field

from ...features import (
    CompiledFeatureContract,
    FeaturePrerequisites,
    ResolvedFeatureTable,
)
from ..contracts import (
    CompiledProblemContract,
    EstimatedBlockRuntimeMetadata,
    ProblemRuntimeMetadata,
)
from ..problem_store import CompiledProblemStore
from ..realization import CompiledRealizationPolicyContract
from ._shared import (
    calibrate_positive_timestamp_delta_seconds,
    resolve_bootstrap_interval_seconds,
    resolve_runtime_interval_seconds,
)
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec

class EstimatedBlockIntervalSource(StrEnum):
    CALIBRATED = "calibrated"
    NOMINAL_CHAIN_RUNTIME = "nominal_chain_runtime"


class EstimatedBlockCalibratedStatistic(StrEnum):
    MEDIAN = "median"
    MEAN = "mean"


class EstimatedBlockCompilerConfig(ProblemCompilerConfig):
    id: str = "estimated_block"
    lookback_interval_source: EstimatedBlockIntervalSource = (
        EstimatedBlockIntervalSource.CALIBRATED
    )
    candidate_interval_source: EstimatedBlockIntervalSource = (
        EstimatedBlockIntervalSource.CALIBRATED
    )
    calibrated_interval_statistic: EstimatedBlockCalibratedStatistic = Field(
        default=EstimatedBlockCalibratedStatistic.MEDIAN
    )


@dataclass(frozen=True, slots=True)
class EstimatedBlockCompiledProblemContract(CompiledProblemContract):
    lookback_interval_source: EstimatedBlockIntervalSource
    candidate_interval_source: EstimatedBlockIntervalSource
    calibrated_interval_statistic: EstimatedBlockCalibratedStatistic
    nominal_block_time_seconds: float | None

    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        lookback_interval_seconds = resolve_bootstrap_interval_seconds(
            source=self.lookback_interval_source,
            recent_block_interval_seconds=recent_block_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
        )
        candidate_interval_seconds = resolve_bootstrap_interval_seconds(
            source=self.candidate_interval_source,
            recent_block_interval_seconds=recent_block_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
        )
        if lookback_interval_seconds is None or candidate_interval_seconds is None:
            return minimum_window
        bootstrap_lookback_steps = _lookback_steps_for_seconds(
            self.lookback_seconds,
            lookback_interval_seconds,
        )
        bootstrap_candidate_count = _candidate_count_for_delay(
            self.max_delay_seconds,
            candidate_interval_seconds,
        )
        return self.feature_prerequisites.history_seconds + math.ceil(
            (
                self.warmup_rows
                + (bootstrap_lookback_steps - 1)
                + self.sample_count
                + bootstrap_candidate_count
            )
            * candidate_interval_seconds
        )

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        store, _ = self.build_capability_store(feature_table)
        return store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> tuple[CompiledProblemStore, EstimatedBlockRuntimeMetadata]:
        calibrated_interval_seconds = calibrate_positive_timestamp_delta_seconds(
            feature_table,
            statistic=self.calibrated_interval_statistic,
            empty_error="Estimated-block compiler requires positive timestamp deltas",
        )
        lookback_interval_seconds = resolve_runtime_interval_seconds(
            source=self.lookback_interval_source,
            calibrated_interval_seconds=calibrated_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
            compiler_label="estimated_block compiler",
            interval_label="lookback",
        )
        candidate_interval_seconds = resolve_runtime_interval_seconds(
            source=self.candidate_interval_source,
            calibrated_interval_seconds=calibrated_interval_seconds,
            nominal_block_time_seconds=self.nominal_block_time_seconds,
            compiler_label="estimated_block compiler",
            interval_label="candidate",
        )
        lookback_steps = _lookback_steps_for_seconds(
            self.lookback_seconds,
            lookback_interval_seconds,
        )
        capability_candidate_count = _candidate_count_for_delay(
            self.max_delay_seconds,
            candidate_interval_seconds,
        )
        store = _build_estimated_block_problem_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_steps=lookback_steps,
            candidate_count=capability_candidate_count,
            requires_post_window_row=self.realization_policy.requires_post_window_row,
        )
        return (
            store,
            EstimatedBlockRuntimeMetadata(
                calibrated_interval_seconds=calibrated_interval_seconds,
                lookback_interval_seconds=lookback_interval_seconds,
                candidate_interval_seconds=candidate_interval_seconds,
                lookback_steps=lookback_steps,
                capability_candidate_count=capability_candidate_count,
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
        if not isinstance(compiler_runtime_metadata, EstimatedBlockRuntimeMetadata):
            raise TypeError("estimated_block requires EstimatedBlockRuntimeMetadata")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if delay_seconds > self.max_delay_seconds:
            raise ValueError(
                "delay_seconds exceeds problem capability: "
                f"{delay_seconds} > {self.max_delay_seconds}"
            )
        candidate_interval_seconds = compiler_runtime_metadata.candidate_interval_seconds
        lookback_steps = compiler_runtime_metadata.lookback_steps
        candidate_count = _candidate_count_for_delay(
            delay_seconds,
            candidate_interval_seconds,
        )
        return _build_estimated_block_problem_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_steps=lookback_steps,
            candidate_count=candidate_count,
            max_candidate_slots=max_candidate_slots,
            requires_post_window_row=self.realization_policy.requires_post_window_row,
        )


def compile_problem(
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    realization_policy: CompiledRealizationPolicyContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
    compiler_config = cast(EstimatedBlockCompilerConfig, problem.compiler)
    nominal_block_time_seconds = (
        None if chain_runtime is None else float(chain_runtime.nominal_block_time_seconds)
    )
    if (
        nominal_block_time_seconds is None
        and (
            compiler_config.lookback_interval_source
            is EstimatedBlockIntervalSource.NOMINAL_CHAIN_RUNTIME
            or compiler_config.candidate_interval_source
            is EstimatedBlockIntervalSource.NOMINAL_CHAIN_RUNTIME
        )
    ):
        raise ValueError(
            "estimated_block compiler requires chain.runtime.nominal_block_time_seconds "
            "for nominal interval resolution"
        )
    return EstimatedBlockCompiledProblemContract(
        compiler_id="estimated_block",
        problem_id=problem.id,
        feature_set_id=feature_contract.feature_set_id,
        feature_family_id=feature_contract.feature_family_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_delay_seconds=problem.max_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
        realization_policy=realization_policy,
        lookback_interval_source=compiler_config.lookback_interval_source,
        candidate_interval_source=compiler_config.candidate_interval_source,
        calibrated_interval_statistic=compiler_config.calibrated_interval_statistic,
        nominal_block_time_seconds=nominal_block_time_seconds,
    )


def _build_estimated_block_problem_store(
    feature_table: ResolvedFeatureTable,
    *,
    feature_prerequisites: FeaturePrerequisites,
    lookback_steps: int,
    candidate_count: int,
    max_candidate_slots: int | None = None,
    requires_post_window_row: bool = False,
) -> CompiledProblemStore:
    if lookback_steps <= 0:
        raise ValueError("lookback_steps must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    timestamps = feature_table.series.timestamps
    if timestamps.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    anchor_candidates = np.arange(timestamps.shape[0], dtype=np.int64)
    context_start_rows = anchor_candidates - lookback_steps + 1
    required_prior_rows = context_start_rows >= 0
    history_ready = required_prior_rows & (
        (timestamps[np.maximum(context_start_rows, 0)] - timestamps[0])
        >= feature_prerequisites.history_seconds
    )
    warmup_ready = required_prior_rows & (context_start_rows >= feature_prerequisites.warmup_rows)
    candidate_end_rows = anchor_candidates + 1 + candidate_count
    future_ready = candidate_end_rows <= timestamps.shape[0]
    post_window_ready = (
        candidate_end_rows < timestamps.shape[0]
        if requires_post_window_row
        else np.ones_like(future_ready, dtype=np.bool_)
    )
    valid_anchor_mask = (
        required_prior_rows & history_ready & warmup_ready & future_ready & post_window_ready
    )
    anchor_rows = anchor_candidates[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    selected_context_starts = context_start_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_ends = candidate_end_rows[anchor_rows].astype(np.int64, copy=False)
    resolved_max_candidate_slots = (
        candidate_count if max_candidate_slots is None else int(max_candidate_slots)
    )
    if resolved_max_candidate_slots <= 0:
        raise ValueError("max_candidate_slots must be positive")
    if candidate_count > resolved_max_candidate_slots:
        raise ValueError("Configured max_candidate_slots is too small for this dataset")

    return CompiledProblemStore(
        feature_matrix=feature_table.feature_matrix,
        log_base_fees=feature_table.series.log_base_fees,
        timestamps=timestamps,
        anchor_rows=anchor_rows,
        context_start_rows=selected_context_starts,
        candidate_end_rows=selected_candidate_ends,
        max_candidate_slots=resolved_max_candidate_slots,
    )

def _lookback_steps_for_seconds(
    lookback_seconds: int,
    effective_block_interval_seconds: float,
) -> int:
    if effective_block_interval_seconds <= 0:
        raise ValueError("effective_block_interval_seconds must be positive")
    return max(1, round(lookback_seconds / effective_block_interval_seconds))


def _candidate_count_for_delay(
    max_delay_seconds: int,
    effective_block_interval_seconds: float,
) -> int:
    if effective_block_interval_seconds <= 0:
        raise ValueError("effective_block_interval_seconds must be positive")
    return max(1, math.floor(max_delay_seconds / effective_block_interval_seconds)) + 1

"""Corpus-calibrated estimated-block problem compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from ...features import (
    CompiledFeatureContract,
    FeaturePrerequisites,
    ResolvedFeatureTable,
)
from ..contracts import CompiledProblemContract, ProblemRuntimeMetadata
from ..problem_store import CompiledProblemStore
from .base import ProblemCompilerConfig, ProblemCompilerSpec
from .registry import register_problem_compiler_spec

_INTERVAL_KEY = "effective_block_interval_seconds"
_LOOKBACK_STEPS_KEY = "lookback_steps"
_CAPABILITY_CANDIDATE_COUNT_KEY = "capability_candidate_count"


class EstimatedBlockCompilerConfig(ProblemCompilerConfig):
    id: Literal["estimated_block"] = "estimated_block"


@dataclass(frozen=True, slots=True)
class EstimatedBlockCompiledProblemContract(CompiledProblemContract):
    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_supported_delay_seconds
        if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
            return minimum_window
        bootstrap_lookback_steps = _lookback_steps_for_seconds(
            self.lookback_seconds,
            recent_block_interval_seconds,
        )
        bootstrap_candidate_count = _candidate_count_for_delay(
            self.max_supported_delay_seconds,
            recent_block_interval_seconds,
        )
        return self.feature_prerequisites.history_seconds + math.ceil(
            (
                self.warmup_rows
                + (bootstrap_lookback_steps - 1)
                + self.sample_count
                + bootstrap_candidate_count
            )
            * recent_block_interval_seconds
        )

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        store, _ = self.build_capability_store(feature_table)
        return store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> tuple[CompiledProblemStore, ProblemRuntimeMetadata]:
        effective_block_interval_seconds = _calibrate_effective_block_interval_seconds(
            feature_table
        )
        lookback_steps = _lookback_steps_for_seconds(
            self.lookback_seconds,
            effective_block_interval_seconds,
        )
        capability_candidate_count = _candidate_count_for_delay(
            self.max_supported_delay_seconds,
            effective_block_interval_seconds,
        )
        store = _build_estimated_block_problem_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_steps=lookback_steps,
            candidate_count=capability_candidate_count,
        )
        return (
            store,
            {
                _INTERVAL_KEY: effective_block_interval_seconds,
                _LOOKBACK_STEPS_KEY: lookback_steps,
                _CAPABILITY_CANDIDATE_COUNT_KEY: capability_candidate_count,
            },
        )

    def build_requested_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        requested_delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        if requested_delay_seconds <= 0:
            raise ValueError("requested_delay_seconds must be positive")
        if requested_delay_seconds > self.max_supported_delay_seconds:
            raise ValueError(
                "requested_delay_seconds exceeds problem capability: "
                f"{requested_delay_seconds} > {self.max_supported_delay_seconds}"
            )
        effective_block_interval_seconds = _runtime_float(
            compiler_runtime_metadata,
            _INTERVAL_KEY,
        )
        lookback_steps = _runtime_int(compiler_runtime_metadata, _LOOKBACK_STEPS_KEY)
        candidate_count = _candidate_count_for_delay(
            requested_delay_seconds,
            effective_block_interval_seconds,
        )
        return _build_estimated_block_problem_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_steps=lookback_steps,
            candidate_count=candidate_count,
            max_candidate_slots=max_candidate_slots,
        )


def compile_problem(
    problem,
    feature_contract: CompiledFeatureContract,
) -> CompiledProblemContract:
    return EstimatedBlockCompiledProblemContract(
        compiler_id="estimated_block",
        problem_id=problem.id,
        feature_set_id=feature_contract.feature_set_id,
        feature_family_id=feature_contract.feature_family_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_supported_delay_seconds=problem.max_supported_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
    )


def _build_estimated_block_problem_store(
    feature_table: ResolvedFeatureTable,
    *,
    feature_prerequisites: FeaturePrerequisites,
    lookback_steps: int,
    candidate_count: int,
    max_candidate_slots: int | None = None,
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
    valid_anchor_mask = required_prior_rows & history_ready & warmup_ready & future_ready
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


def _calibrate_effective_block_interval_seconds(feature_table: ResolvedFeatureTable) -> float:
    deltas = np.diff(feature_table.series.timestamps.astype(np.int64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError("Estimated-block compiler requires positive timestamp deltas")
    return float(np.median(positive_deltas))


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


def _runtime_float(metadata: ProblemRuntimeMetadata, key: str) -> float:
    try:
        value = metadata[key]
    except KeyError as exc:
        raise ValueError(f"Missing compiler runtime metadata: {key}") from exc
    return float(cast(int | float | str, value))


def _runtime_int(metadata: ProblemRuntimeMetadata, key: str) -> int:
    try:
        value = metadata[key]
    except KeyError as exc:
        raise ValueError(f"Missing compiler runtime metadata: {key}") from exc
    return int(cast(int | float | str, value))


register_problem_compiler_spec(
    ProblemCompilerSpec[EstimatedBlockCompilerConfig](
        id="estimated_block",
        config_type=EstimatedBlockCompilerConfig,
        compile_problem=cast(object, compile_problem),
    )
)

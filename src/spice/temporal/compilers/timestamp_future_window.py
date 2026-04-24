"""Timestamp-bounded problem compiler with fixed current-row candidate semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic import SerializeAsAny, field_validator

from ...features import (
    CompiledFeatureContract,
    ResolvedFeatureTable,
)
from ...modeling.families.base import ConfigModel
from ..contracts import CompiledProblemContract
from ..problem_store import CompiledProblemStore
from ..realization import CompiledRealizationPolicyContract
from ._shared import (
    build_timestamp_window_store,
    resolve_bootstrap_interval_seconds,
    resolve_interval_estimator_seconds,
)
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec


class TimestampFutureWindowIntervalEstimatorConfig(ConfigModel):
    id: str


class TimestampFutureWindowNominalIntervalEstimatorConfig(
    TimestampFutureWindowIntervalEstimatorConfig
):
    id: str = "nominal"


class TimestampFutureWindowRecentDeltasIntervalEstimatorConfig(
    TimestampFutureWindowIntervalEstimatorConfig
):
    id: str = "recent_deltas"


class TimestampFutureWindowCompilerConfig(ProblemCompilerConfig):
    id: str = "timestamp_future_window"
    action_interval_estimator: SerializeAsAny[TimestampFutureWindowIntervalEstimatorConfig]

    @field_validator("action_interval_estimator", mode="before")
    @classmethod
    def validate_action_interval_estimator(
        cls,
        value: object,
    ) -> TimestampFutureWindowIntervalEstimatorConfig:
        if isinstance(value, TimestampFutureWindowIntervalEstimatorConfig):
            raw_payload = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            raw_payload = dict(value)
        else:
            raise TypeError("timestamp_future_window.action_interval_estimator must be a mapping")
        estimator_id = raw_payload.get("id")
        if estimator_id == "nominal":
            return TimestampFutureWindowNominalIntervalEstimatorConfig.model_validate(raw_payload)
        if estimator_id == "recent_deltas":
            return TimestampFutureWindowRecentDeltasIntervalEstimatorConfig.model_validate(
                raw_payload
            )
        known = ", ".join(sorted(("nominal", "recent_deltas")))
        raise ValueError(
            "timestamp_future_window.action_interval_estimator.id must be one of: "
            f"{known}"
        )


@dataclass(frozen=True, slots=True)
class TimestampFutureWindowRuntimeMetadata:
    action_interval_estimator_id: str
    action_interval_seconds: float
    capability_action_count: int


@dataclass(frozen=True, slots=True)
class TimestampFutureWindowCompiledProblemContract(CompiledProblemContract):
    action_interval_estimator: TimestampFutureWindowIntervalEstimatorConfig
    nominal_block_time_seconds: float | None

    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        bootstrap_interval_seconds = resolve_bootstrap_interval_seconds(
            estimator=self.action_interval_estimator,
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
        action_interval_seconds = resolve_interval_estimator_seconds(
            estimator=self.action_interval_estimator,
            feature_table=feature_table,
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
            max_candidate_slots=capability_action_count,
            requires_post_window_row=self.realization_policy.requires_post_window_row,
        )
        return (
            store,
            TimestampFutureWindowRuntimeMetadata(
                action_interval_estimator_id=self.action_interval_estimator.id,
                action_interval_seconds=action_interval_seconds,
                capability_action_count=capability_action_count,
            ),
        )

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: object,
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
            max_candidate_slots=max_candidate_slots,
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
        compiler_config.action_interval_estimator.id == "nominal"
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
        realization_policy=replace(realization_policy, requires_post_window_row=True),
        action_interval_estimator=compiler_config.action_interval_estimator,
        nominal_block_time_seconds=nominal_block_time_seconds,
    )


def runtime_metadata_payload(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, TimestampFutureWindowRuntimeMetadata):
        raise TypeError("timestamp_future_window requires TimestampFutureWindowRuntimeMetadata")
    return {
        "action_interval_estimator_id": metadata.action_interval_estimator_id,
        "action_interval_seconds": metadata.action_interval_seconds,
        "capability_action_count": metadata.capability_action_count,
    }


def runtime_metadata_from_payload(
    payload: Mapping[str, object],
) -> TimestampFutureWindowRuntimeMetadata:
    raw_payload = dict(payload)
    return TimestampFutureWindowRuntimeMetadata(
        action_interval_estimator_id=_str_payload(raw_payload, "action_interval_estimator_id"),
        action_interval_seconds=_float_payload(raw_payload, "action_interval_seconds"),
        capability_action_count=_int_payload(raw_payload, "capability_action_count"),
    )


def _float_payload(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid float runtime metadata field: {key}")
    return float(value)


def _int_payload(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid integer runtime metadata field: {key}")
    return int(value)


def _str_payload(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Invalid string runtime metadata field: {key}")
    return value


def _action_count_for_delay(
    max_delay_seconds: int,
    interval_seconds: float,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return max(1, math.floor(max_delay_seconds / interval_seconds)) + 1

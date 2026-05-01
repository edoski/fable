"""Timestamp-bounded problem compiler with fixed current-row candidate semantics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import SerializeAsAny, field_validator

from ...core.specs import lookup_local_spec, require_mapping_id
from ...features import (
    CompiledFeatureContract,
    ResolvedFeatureTable,
)
from ...modeling.families.base import ConfigModel
from ..contracts import CompiledProblemContract
from ..execution_policy import CompiledExecutionPolicyContract
from ..problem_store import CompiledProblemStore
from ._shared import (
    build_timestamp_window_store,
    summarize_positive_timestamp_delta_seconds,
)
from .base import ProblemCompilerConfig

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec


class ObservedTimeWindowSlotSpacingConfig(ConfigModel):
    id: str


class ObservedTimeWindowNominalSlotSpacingConfig(
    ObservedTimeWindowSlotSpacingConfig
):
    id: str = "nominal"


class ObservedTimeWindowRecentMedianSlotSpacingConfig(
    ObservedTimeWindowSlotSpacingConfig
):
    id: str = "recent_median"


@dataclass(frozen=True, slots=True)
class _SlotSpacingSpec:
    config_type: type[ObservedTimeWindowSlotSpacingConfig]
    requires_nominal_runtime: bool
    resolve_slot_spacing_seconds: Callable[
        [ResolvedFeatureTable, float | None],
        float,
    ]
    resolve_bootstrap_interval_seconds: Callable[
        [float | None, float | None],
        float | None,
    ]


def _resolve_nominal_slot_spacing_seconds(
    feature_table: ResolvedFeatureTable,
    nominal_block_time_seconds: float | None,
) -> float:
    del feature_table
    if nominal_block_time_seconds is None or nominal_block_time_seconds <= 0:
        raise ValueError(
            "observed_time_window requires nominal block time "
            "for slot spacing resolution"
        )
    return nominal_block_time_seconds


def _resolve_recent_median_slot_spacing_seconds(
    feature_table: ResolvedFeatureTable,
    nominal_block_time_seconds: float | None,
) -> float:
    del nominal_block_time_seconds
    return summarize_positive_timestamp_delta_seconds(
        feature_table,
        statistic="median",
        empty_error="observed_time_window requires positive timestamp deltas",
    )


def _resolve_nominal_bootstrap_interval_seconds(
    recent_block_interval_seconds: float | None,
    nominal_block_time_seconds: float | None,
) -> float | None:
    del recent_block_interval_seconds
    return nominal_block_time_seconds


def _resolve_recent_median_bootstrap_interval_seconds(
    recent_block_interval_seconds: float | None,
    nominal_block_time_seconds: float | None,
) -> float | None:
    del nominal_block_time_seconds
    if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
        return None
    return recent_block_interval_seconds


_SLOT_SPACING_SPECS: dict[str, _SlotSpacingSpec] = {
    "nominal": _SlotSpacingSpec(
        config_type=ObservedTimeWindowNominalSlotSpacingConfig,
        requires_nominal_runtime=True,
        resolve_slot_spacing_seconds=_resolve_nominal_slot_spacing_seconds,
        resolve_bootstrap_interval_seconds=_resolve_nominal_bootstrap_interval_seconds,
    ),
    "recent_median": _SlotSpacingSpec(
        config_type=ObservedTimeWindowRecentMedianSlotSpacingConfig,
        requires_nominal_runtime=False,
        resolve_slot_spacing_seconds=_resolve_recent_median_slot_spacing_seconds,
        resolve_bootstrap_interval_seconds=_resolve_recent_median_bootstrap_interval_seconds,
    ),
}


def _slot_spacing_spec(
    slot_spacing_id: str,
) -> _SlotSpacingSpec:
    return lookup_local_spec(
        _SLOT_SPACING_SPECS,
        slot_spacing_id,
        "observed_time_window.slot_spacing.id",
    )


class ObservedTimeWindowCompilerConfig(ProblemCompilerConfig):
    id: str = "observed_time_window"
    slot_spacing: SerializeAsAny[ObservedTimeWindowSlotSpacingConfig]

    @field_validator("slot_spacing", mode="before")
    @classmethod
    def validate_slot_spacing(
        cls,
        value: object,
    ) -> ObservedTimeWindowSlotSpacingConfig:
        if isinstance(value, ObservedTimeWindowSlotSpacingConfig):
            raw_payload = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            raw_payload = dict(value)
        else:
            raise ValueError("observed_time_window.slot_spacing must be a mapping")
        slot_spacing_id = require_mapping_id(
            raw_payload,
            "observed_time_window.slot_spacing.id",
        )
        return _slot_spacing_spec(slot_spacing_id).config_type.model_validate(
            raw_payload
        )


@dataclass(frozen=True, slots=True)
class ObservedTimeWindowRuntimeMetadata:
    slot_spacing_id: str
    slot_spacing_seconds: float
    capability_action_count: int


@dataclass(frozen=True, slots=True)
class ObservedTimeWindowCompiledProblemContract(CompiledProblemContract):
    slot_spacing: ObservedTimeWindowSlotSpacingConfig
    nominal_block_time_seconds: float | None

    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        minimum_window = self.required_history_seconds + self.max_delay_seconds
        bootstrap_interval_seconds = _slot_spacing_spec(
            self.slot_spacing.id
        ).resolve_bootstrap_interval_seconds(
            recent_block_interval_seconds,
            self.nominal_block_time_seconds,
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
    ) -> tuple[CompiledProblemStore, ObservedTimeWindowRuntimeMetadata]:
        slot_spacing_seconds = _slot_spacing_spec(
            self.slot_spacing.id
        ).resolve_slot_spacing_seconds(
            feature_table,
            self.nominal_block_time_seconds,
        )
        capability_action_count = _action_count_for_delay(
            self.max_delay_seconds,
            slot_spacing_seconds,
        )
        store = build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=self.max_delay_seconds,
            max_candidate_slots=capability_action_count,
            requires_post_window_row=self.execution_policy.requires_post_window_row,
        )
        return (
            store,
            ObservedTimeWindowRuntimeMetadata(
                slot_spacing_id=self.slot_spacing.id,
                slot_spacing_seconds=slot_spacing_seconds,
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
        if not isinstance(compiler_runtime_metadata, ObservedTimeWindowRuntimeMetadata):
            raise TypeError("observed_time_window requires ObservedTimeWindowRuntimeMetadata")
        if max_candidate_slots != compiler_runtime_metadata.capability_action_count:
            raise ValueError(
                "artifact action width does not match observed_time_window runtime metadata: "
                f"{max_candidate_slots} != "
                f"{compiler_runtime_metadata.capability_action_count}"
            )
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
            requires_post_window_row=self.execution_policy.requires_post_window_row,
        )


def compile_problem(
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    execution_policy: CompiledExecutionPolicyContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
    compiler_config = ObservedTimeWindowCompilerConfig.model_validate(problem.compiler)
    nominal_block_time_seconds = (
        None if chain_runtime is None else float(chain_runtime.nominal_block_time_seconds)
    )
    slot_spacing_spec = _slot_spacing_spec(compiler_config.slot_spacing.id)
    if slot_spacing_spec.requires_nominal_runtime and (
        nominal_block_time_seconds is None or nominal_block_time_seconds <= 0
    ):
        raise ValueError(
            "observed_time_window requires chain.runtime.nominal_block_time_seconds "
            "for nominal slot-spacing resolution"
        )
    return ObservedTimeWindowCompiledProblemContract(
        compiler_id="observed_time_window",
        problem_id=problem.id,
        features_id=feature_contract.features_id,
        lookback_seconds=problem.lookback_seconds,
        sample_count=problem.sample_count,
        max_delay_seconds=problem.max_delay_seconds,
        feature_prerequisites=feature_contract.feature_prerequisites,
        execution_policy=execution_policy,
        slot_spacing=compiler_config.slot_spacing,
        nominal_block_time_seconds=nominal_block_time_seconds,
    )


def runtime_metadata_payload(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, ObservedTimeWindowRuntimeMetadata):
        raise TypeError("observed_time_window requires ObservedTimeWindowRuntimeMetadata")
    return {
        "slot_spacing_id": metadata.slot_spacing_id,
        "slot_spacing_seconds": metadata.slot_spacing_seconds,
        "capability_action_count": metadata.capability_action_count,
    }


def runtime_metadata_from_payload(
    payload: Mapping[str, object],
) -> ObservedTimeWindowRuntimeMetadata:
    raw_payload = dict(payload)
    return ObservedTimeWindowRuntimeMetadata(
        slot_spacing_id=_str_payload(raw_payload, "slot_spacing_id"),
        slot_spacing_seconds=_float_payload(raw_payload, "slot_spacing_seconds"),
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

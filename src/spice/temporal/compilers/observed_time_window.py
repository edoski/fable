"""Timestamp-bounded problem compiler with fixed current-row candidate semantics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import SerializeAsAny, field_validator

from ...core.specs import lookup_local_spec, owner_payload_id, validate_owner_config
from ...core.validation import validate_path_segment
from ...features import (
    CompiledFeatureContract,
    ResolvedFeatureTable,
)
from ...modeling.families.base import ConfigModel
from ..capability import TemporalCapability
from ..contracts import CompiledProblemContract, TemporalCapabilityStore
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

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="observed_time_window.slot_spacing.id")


class ObservedTimeWindowNominalSlotSpacingConfig(
    ObservedTimeWindowSlotSpacingConfig
):
    id: str = "nominal"

    @field_validator("id")
    @classmethod
    def validate_nominal_id(cls, value: str) -> str:
        value = ObservedTimeWindowSlotSpacingConfig.validate_id(value)
        if value != "nominal":
            raise ValueError("observed_time_window.slot_spacing.id must be nominal")
        return value


class ObservedTimeWindowRecentMedianSlotSpacingConfig(
    ObservedTimeWindowSlotSpacingConfig
):
    id: str = "recent_median"

    @field_validator("id")
    @classmethod
    def validate_recent_median_id(cls, value: str) -> str:
        value = ObservedTimeWindowSlotSpacingConfig.validate_id(value)
        if value != "recent_median":
            raise ValueError("observed_time_window.slot_spacing.id must be recent_median")
        return value


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

    @field_validator("id")
    @classmethod
    def validate_observed_time_window_id(cls, value: str) -> str:
        value = ProblemCompilerConfig.validate_id(value)
        if value != "observed_time_window":
            raise ValueError("problem.compiler.id must be observed_time_window")
        return value

    @field_validator("slot_spacing", mode="before")
    @classmethod
    def validate_slot_spacing(
        cls,
        value: object,
    ) -> ObservedTimeWindowSlotSpacingConfig:
        raw_payload, slot_spacing_id = owner_payload_id(
            value,
            owner="observed_time_window.slot_spacing",
            config_type=ObservedTimeWindowSlotSpacingConfig,
            id_label="observed_time_window.slot_spacing.id",
        )
        spec = _slot_spacing_spec(slot_spacing_id)
        if isinstance(value, spec.config_type):
            return value
        return validate_owner_config(raw_payload, spec.config_type)


@dataclass(frozen=True, slots=True)
class ObservedTimeWindowRuntimeMetadata:
    slot_spacing_id: str
    slot_spacing_seconds: float


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
        return self.build_capability_store(feature_table).store.n_samples

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> TemporalCapabilityStore:
        slot_spacing_seconds = _slot_spacing_spec(
            self.slot_spacing.id
        ).resolve_slot_spacing_seconds(
            feature_table,
            self.nominal_block_time_seconds,
        )
        action_width = _action_count_for_delay(
            self.max_delay_seconds,
            slot_spacing_seconds,
        )
        store = build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=self.max_delay_seconds,
            max_candidate_slots=action_width,
            requires_post_window_row=self.execution_policy.requires_post_window_row,
        )
        return TemporalCapabilityStore(
            store=store,
            capability=TemporalCapability(
                compiler_id=self.compiler_id,
                max_delay_seconds=self.max_delay_seconds,
                action_width=action_width,
                compiler_runtime_metadata=ObservedTimeWindowRuntimeMetadata(
                    slot_spacing_id=self.slot_spacing.id,
                    slot_spacing_seconds=slot_spacing_seconds,
                ),
            ),
        )

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        capability: TemporalCapability,
    ) -> CompiledProblemStore:
        if capability.compiler_id != self.compiler_id:
            raise ValueError(
                "temporal capability compiler does not match problem compiler: "
                f"{capability.compiler_id} != {self.compiler_id}"
            )
        if capability.max_delay_seconds != self.max_delay_seconds:
            raise ValueError(
                "temporal capability delay does not match problem capability: "
                f"{capability.max_delay_seconds} != {self.max_delay_seconds}"
            )
        metadata = capability.compiler_runtime_metadata
        if not isinstance(metadata, ObservedTimeWindowRuntimeMetadata):
            raise TypeError("observed_time_window requires ObservedTimeWindowRuntimeMetadata")
        expected_action_width = _action_count_for_delay(
            capability.max_delay_seconds,
            metadata.slot_spacing_seconds,
        )
        if capability.action_width != expected_action_width:
            raise ValueError(
                "temporal capability action width does not match observed_time_window metadata: "
                f"{capability.action_width} != {expected_action_width}"
            )
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if delay_seconds > capability.max_delay_seconds:
            raise ValueError(
                "delay_seconds exceeds problem capability: "
                f"{delay_seconds} > {capability.max_delay_seconds}"
            )
        return build_timestamp_window_store(
            feature_table,
            feature_prerequisites=self.feature_prerequisites,
            lookback_seconds=self.lookback_seconds,
            delay_seconds=delay_seconds,
            max_candidate_slots=capability.action_width,
            requires_post_window_row=self.execution_policy.requires_post_window_row,
        )


def compile_problem(
    problem: ProblemSpec,
    compiler_config: ObservedTimeWindowCompilerConfig,
    feature_contract: CompiledFeatureContract,
    execution_policy: CompiledExecutionPolicyContract,
    chain_runtime: ChainRuntimeSpec | None,
) -> CompiledProblemContract:
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
    }


def runtime_metadata_from_payload(
    payload: Mapping[str, object],
) -> ObservedTimeWindowRuntimeMetadata:
    raw_payload = dict(payload)
    return ObservedTimeWindowRuntimeMetadata(
        slot_spacing_id=_str_payload(raw_payload, "slot_spacing_id"),
        slot_spacing_seconds=_float_payload(raw_payload, "slot_spacing_seconds"),
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

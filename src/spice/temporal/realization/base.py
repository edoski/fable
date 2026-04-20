"""Problem-owned realization policy seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from ...core.closed_dispatch import (
    config_payload_and_id,
    unknown_id_error,
    validate_path_segment,
)
from ...modeling.families.base import ConfigModel
from ...semantics import RealizationPolicySemantics
from ..problem_store import CompiledProblemStore

IntVector = NDArray[np.int64]
BoolMatrix = NDArray[np.bool_]
BoolVector = NDArray[np.bool_]
FloatVector = NDArray[np.float32]


class RealizationPolicyConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.realization_policy.id")


@dataclass(frozen=True, slots=True)
class PreparedSupervisedRealizationTargets:
    candidate_mask: BoolMatrix
    optimum_offsets: IntVector
    optimum_log_fees: FloatVector


@dataclass(frozen=True, slots=True)
class RealizedSelectionBatch:
    realized_rows: IntVector
    baseline_rows: IntVector
    optimum_rows: IntVector
    requested_offsets: IntVector
    resolved_offsets: IntVector
    overflow_mask: BoolVector


PrepareSupervisedTargetsFn = Callable[
    [CompiledProblemStore, IntVector],
    PreparedSupervisedRealizationTargets,
]
RealizeSelectionsFn = Callable[
    [CompiledProblemStore, Sequence[int], IntVector, IntVector],
    RealizedSelectionBatch,
]


@dataclass(frozen=True, slots=True)
class CompiledRealizationPolicyContract:
    realization_policy_id: str
    prepare_supervised_targets_fn: PrepareSupervisedTargetsFn
    realize_selections_fn: RealizeSelectionsFn

    @property
    def semantics(self) -> RealizationPolicySemantics:
        return RealizationPolicySemantics(
            realization_policy_id=self.realization_policy_id,
        )

    def prepare_supervised_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedSupervisedRealizationTargets:
        return self.prepare_supervised_targets_fn(store, sample_indices)

    def realize_selections(
        self,
        store: CompiledProblemStore,
        decoded_offsets: Sequence[int],
        sample_indices: IntVector,
        selected_positions: IntVector,
    ) -> RealizedSelectionBatch:
        return self.realize_selections_fn(
            store,
            decoded_offsets,
            sample_indices,
            selected_positions,
        )


@dataclass(frozen=True, slots=True)
class RealizationPolicySpec:
    id: str
    config_type: type[RealizationPolicyConfig]
    compile: Callable[[RealizationPolicyConfig], CompiledRealizationPolicyContract]


def _compile_strict_deadline_miss(
    config: RealizationPolicyConfig,
) -> CompiledRealizationPolicyContract:
    from .strict_deadline_miss import StrictDeadlineMissConfig, compile_realization_policy

    return compile_realization_policy(StrictDeadlineMissConfig.model_validate(config))


def realization_policy_spec(policy_id: str) -> RealizationPolicySpec:
    if policy_id == "strict_deadline_miss":
        from .strict_deadline_miss import StrictDeadlineMissConfig

        return RealizationPolicySpec(
            id="strict_deadline_miss",
            config_type=StrictDeadlineMissConfig,
            compile=_compile_strict_deadline_miss,
        )
    raise unknown_id_error(
        field_name="problem.realization_policy.id",
        component_id=policy_id,
        known_ids=("strict_deadline_miss",),
    )


def coerce_realization_policy_config(
    payload: Mapping[str, object] | RealizationPolicyConfig,
) -> RealizationPolicyConfig:
    raw_payload, policy_id = config_payload_and_id(
        payload,
        config_type=RealizationPolicyConfig,
        field_name="problem.realization_policy.id",
        mapping_label="problem.realization_policy",
    )
    spec = realization_policy_spec(policy_id)
    return spec.config_type.model_validate(raw_payload)


def compile_realization_policy_contract(
    config: RealizationPolicyConfig,
) -> CompiledRealizationPolicyContract:
    return realization_policy_spec(config.id).compile(config)

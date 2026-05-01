"""Problem-owned execution policy seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from ...core.errors import ConfigResolutionError
from ...core.specs import lookup_local_spec, require_mapping_id, require_spec_config
from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel
from ...semantics import ExecutionPolicySemantics
from ..problem_store import CompiledProblemStore
from ..semantics import BaselineRowMode

IntVector = NDArray[np.int64]
BoolMatrix = NDArray[np.bool_]
BoolVector = NDArray[np.bool_]
FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]


class ExecutionPolicyConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.execution_policy.id")


@dataclass(frozen=True, slots=True)
class PreparedActionSpace:
    action_mask: BoolMatrix


@dataclass(frozen=True, slots=True)
class PreparedSupervisedExecutionTargets:
    action_mask: BoolMatrix
    candidate_log_fees: FloatMatrix
    optimum_offsets: IntVector
    optimum_log_fees: FloatVector
    baseline_candidate_indices: IntVector


@dataclass(frozen=True, slots=True)
class RealizedSelectionBatch:
    realized_rows: IntVector
    baseline_rows: IntVector
    optimum_rows: IntVector
    requested_offsets: IntVector
    resolved_offsets: IntVector
    overflow_mask: BoolVector


class DecodedOffsetBatch(Protocol):
    def __len__(self) -> int: ...

    def select(self, sample_positions: IntVector) -> IntVector: ...


PrepareSupervisedTargetsFn = Callable[
    [CompiledProblemStore, IntVector],
    PreparedSupervisedExecutionTargets,
]
PrepareActionSpaceFn = Callable[
    [CompiledProblemStore, IntVector],
    PreparedActionSpace,
]
RealizeSelectionsFn = Callable[
    [CompiledProblemStore, DecodedOffsetBatch, IntVector, IntVector],
    RealizedSelectionBatch,
]


@dataclass(frozen=True, slots=True)
class CompiledExecutionPolicyContract:
    execution_policy_id: str
    baseline_row_mode: BaselineRowMode
    requires_post_window_row: bool
    prepare_action_space_fn: PrepareActionSpaceFn
    prepare_supervised_targets_fn: PrepareSupervisedTargetsFn
    realize_selections_fn: RealizeSelectionsFn

    @property
    def semantics(self) -> ExecutionPolicySemantics:
        return ExecutionPolicySemantics(
            execution_policy_id=self.execution_policy_id,
            baseline_row_mode=self.baseline_row_mode.value,
        )

    def prepare_supervised_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedSupervisedExecutionTargets:
        return self.prepare_supervised_targets_fn(store, sample_indices)

    def prepare_action_space(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedActionSpace:
        return self.prepare_action_space_fn(store, sample_indices)

    def realize_selections(
        self,
        store: CompiledProblemStore,
        decoded_offsets: DecodedOffsetBatch,
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
class ExecutionPolicySpec:
    config_type: type[ExecutionPolicyConfig]
    compile_contract: Callable[[Any], CompiledExecutionPolicyContract]


def _compile_strict_deadline_miss(
    config: Any,
) -> CompiledExecutionPolicyContract:
    from .strict_deadline_miss import compile_execution_policy

    return compile_execution_policy(config)


def _execution_policy_specs() -> dict[str, ExecutionPolicySpec]:
    from .strict_deadline_miss import StrictDeadlineMissConfig

    return {
        "strict_deadline_miss": ExecutionPolicySpec(
            config_type=StrictDeadlineMissConfig,
            compile_contract=_compile_strict_deadline_miss,
        ),
    }


def execution_policy_spec(policy_id: str) -> ExecutionPolicySpec:
    return lookup_local_spec(
        _execution_policy_specs(),
        policy_id,
        "problem.execution_policy.id",
    )


def coerce_execution_policy_config(
    payload: Mapping[str, object] | ExecutionPolicyConfig,
) -> ExecutionPolicyConfig:
    if isinstance(payload, ExecutionPolicyConfig):
        raw_payload = payload.model_dump(mode="json")
        policy_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        policy_id = require_mapping_id(raw_payload, "problem.execution_policy.id")
    else:
        raise ConfigResolutionError(
            "problem.execution_policy must be a mapping or config model"
        )
    return execution_policy_spec(policy_id).config_type.model_validate(raw_payload)


def compile_execution_policy_contract(
    config: ExecutionPolicyConfig,
) -> CompiledExecutionPolicyContract:
    spec = execution_policy_spec(config.id)
    concrete_config = require_spec_config(config, spec.config_type, "execution policy config")
    return spec.compile_contract(concrete_config)

"""Problem-owned execution policy seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from ...core.config_model import ConfigModel
from ...core.specs import (
    lookup_local_spec,
    owner_payload_id,
    require_spec_config,
    validate_owner_config,
)
from ...core.validation import validate_path_segment
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
    sample_indices: IntVector
    max_candidate_slots: int
    action_mask: BoolMatrix

    def __post_init__(self) -> None:
        sample_indices = self.sample_indices.astype(np.int64, copy=False)
        action_mask = self.action_mask.astype(np.bool_, copy=False)
        max_candidate_slots = int(self.max_candidate_slots)
        if sample_indices.ndim != 1:
            raise ValueError("Action Space sample_indices must be a one-dimensional array")
        if max_candidate_slots <= 0:
            raise ValueError("Action Space max_candidate_slots must be positive")
        if action_mask.shape != (int(sample_indices.shape[0]), max_candidate_slots):
            raise ValueError(
                "Action Space action_mask must match sample count and action width"
            )
        object.__setattr__(self, "sample_indices", sample_indices)
        object.__setattr__(self, "max_candidate_slots", max_candidate_slots)
        object.__setattr__(self, "action_mask", action_mask)


@dataclass(frozen=True, slots=True)
class PreparedSupervisedExecutionTargets:
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
    [CompiledProblemStore, PreparedActionSpace],
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
        action_space: PreparedActionSpace,
    ) -> PreparedSupervisedExecutionTargets:
        return self.prepare_supervised_targets_fn(store, action_space)

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
    payload: object,
) -> ExecutionPolicyConfig:
    raw_payload, policy_id = owner_payload_id(
        payload,
        owner="problem.execution_policy",
        config_type=ExecutionPolicyConfig,
        id_label="problem.execution_policy.id",
    )
    spec = execution_policy_spec(policy_id)
    if isinstance(payload, spec.config_type):
        return payload
    return validate_owner_config(raw_payload, spec.config_type)


def compile_execution_policy_contract(
    config: ExecutionPolicyConfig,
) -> CompiledExecutionPolicyContract:
    spec = execution_policy_spec(config.id)
    concrete_config = require_spec_config(config, spec.config_type, "execution policy config")
    return spec.compile_contract(concrete_config)

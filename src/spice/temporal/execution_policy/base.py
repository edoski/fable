"""Problem-owned execution policy seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from ...core.config_model import ConfigModel
from ...core.specs import lookup_local_spec, owner_payload, validate_owner_config
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
        if action_mask.shape[0] > 0 and not bool(np.all(action_mask.any(axis=1))):
            raise ValueError("Action Space action_mask must allow at least one action per sample")
        object.__setattr__(self, "sample_indices", sample_indices)
        object.__setattr__(self, "max_candidate_slots", max_candidate_slots)
        object.__setattr__(self, "action_mask", action_mask)

    def validate_selected_positions(self, selected_positions: IntVector) -> IntVector:
        resolved = np.asarray(selected_positions)
        if resolved.ndim != 1:
            raise ValueError("selected_positions must be one-dimensional")
        if resolved.size == 0:
            raise ValueError("selected_positions must be non-empty")
        if not np.issubdtype(resolved.dtype, np.integer):
            raise ValueError("selected_positions must be integer indices")
        resolved = resolved.astype(np.int64, copy=False)
        sample_count = int(self.sample_indices.shape[0])
        if bool(np.any(resolved < 0)) or bool(np.any(resolved >= sample_count)):
            raise ValueError("selected_positions are outside sample_indices")
        return resolved

    def validate_requested_offsets(
        self,
        selected_positions: IntVector,
        requested_offsets: IntVector,
    ) -> IntVector:
        resolved_offsets = np.asarray(requested_offsets)
        if resolved_offsets.ndim != 1:
            raise ValueError("decoded_offsets must be one-dimensional")
        if not np.issubdtype(resolved_offsets.dtype, np.integer):
            raise ValueError("decoded_offsets must be integer offsets")
        resolved_offsets = resolved_offsets.astype(np.int64, copy=False)
        if resolved_offsets.shape != selected_positions.shape:
            raise ValueError("decoded_offsets must align with selected_positions")
        if bool(np.any(resolved_offsets < 0)):
            raise ValueError("decoded_offsets must be non-negative")
        if bool(np.any(resolved_offsets >= self.max_candidate_slots)):
            raise ValueError("decoded_offsets must be smaller than max_candidate_slots")
        if not bool(np.all(self.action_mask[selected_positions, resolved_offsets])):
            raise ValueError("decoded_offsets contain unavailable actions")
        return resolved_offsets


@dataclass(frozen=True, slots=True)
class PreparedTemporalOutcomeFacts:
    action_outcome_rows: NDArray[np.int64]
    action_outcome_log_fees: FloatMatrix
    reachable_action_mask: BoolMatrix
    baseline_rows: IntVector
    overflow_mask: BoolMatrix


@dataclass(frozen=True, slots=True)
class PreparedTemporalFacts:
    action_space: PreparedActionSpace
    outcome_facts: PreparedTemporalOutcomeFacts

    def __post_init__(self) -> None:
        sample_count = int(self.action_space.sample_indices.shape[0])
        action_shape = (sample_count, self.action_space.max_candidate_slots)
        if self.outcome_facts.action_outcome_rows.shape != action_shape:
            raise ValueError("temporal outcome rows must match Action Space shape")
        if self.outcome_facts.action_outcome_log_fees.shape != action_shape:
            raise ValueError("temporal outcome log fees must match Action Space shape")
        if self.outcome_facts.reachable_action_mask.shape != action_shape:
            raise ValueError("temporal reachable action mask must match Action Space shape")
        if self.outcome_facts.overflow_mask.shape != action_shape:
            raise ValueError("temporal overflow mask must match Action Space shape")
        if self.outcome_facts.baseline_rows.shape != (sample_count,):
            raise ValueError("temporal baseline rows must match sample count")
        if self.outcome_facts.reachable_action_mask.shape[0] > 0 and not bool(
            np.all(self.outcome_facts.reachable_action_mask.any(axis=1))
        ):
            raise ValueError("temporal outcome facts must keep one reachable action per sample")
        reachable_actions_available = np.all(
            self.outcome_facts.reachable_action_mask <= self.action_space.action_mask
        )
        if not bool(reachable_actions_available):
            raise ValueError(
                "temporal reachable action mask must be a subset of Action Space action mask"
            )


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


PrepareOutcomeFactsFn = Callable[
    [CompiledProblemStore, PreparedActionSpace],
    PreparedTemporalOutcomeFacts,
]
PrepareActionSpaceFn = Callable[
    [CompiledProblemStore, IntVector],
    PreparedActionSpace,
]
RealizeSelectionsFn = Callable[
    [CompiledProblemStore, DecodedOffsetBatch, PreparedActionSpace, IntVector],
    RealizedSelectionBatch,
]


@dataclass(frozen=True, slots=True)
class CompiledExecutionPolicyContract:
    execution_policy_id: str
    baseline_row_mode: BaselineRowMode
    requires_post_window_row: bool
    prepare_action_space_fn: PrepareActionSpaceFn
    prepare_outcome_facts_fn: PrepareOutcomeFactsFn
    realize_selections_fn: RealizeSelectionsFn

    @property
    def semantics(self) -> ExecutionPolicySemantics:
        return ExecutionPolicySemantics(
            execution_policy_id=self.execution_policy_id,
            baseline_row_mode=self.baseline_row_mode.value,
        )

    def prepare_action_space(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedActionSpace:
        action_space = self.prepare_action_space_fn(store, sample_indices)
        resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
        if not np.array_equal(action_space.sample_indices, resolved_sample_indices):
            raise ValueError("prepared Action Space sample_indices do not match request")
        if action_space.max_candidate_slots != int(store.max_candidate_slots):
            raise ValueError("prepared Action Space action width does not match store")
        expected_shape = (
            int(resolved_sample_indices.shape[0]),
            int(store.max_candidate_slots),
        )
        if action_space.action_mask.shape != expected_shape:
            raise ValueError(
                "prepared Action Space action_mask shape does not match sample count "
                "and action width"
            )
        return action_space

    def prepare_temporal_facts(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedTemporalFacts:
        action_space = self.prepare_action_space(store, sample_indices)
        return PreparedTemporalFacts(
            action_space=action_space,
            outcome_facts=self.prepare_outcome_facts_fn(store, action_space),
        )

    def realize_selections(
        self,
        store: CompiledProblemStore,
        decoded_offsets: DecodedOffsetBatch,
        action_space: PreparedActionSpace,
        selected_positions: IntVector,
    ) -> RealizedSelectionBatch:
        if len(decoded_offsets) != int(action_space.sample_indices.shape[0]):
            raise ValueError("decoded_offsets must align with sample_indices")
        resolved_positions = action_space.validate_selected_positions(selected_positions)
        requested_offsets = decoded_offsets.select(resolved_positions)
        action_space.validate_requested_offsets(resolved_positions, requested_offsets)
        return self.realize_selections_fn(
            store,
            decoded_offsets,
            action_space,
            resolved_positions,
        )


_SUPPORTED_EXECUTION_POLICY_IDS = frozenset({"strict_deadline_miss"})


def _require_execution_policy_id(policy_id: str) -> str:
    return lookup_local_spec(
        {policy_id: policy_id for policy_id in _SUPPORTED_EXECUTION_POLICY_IDS},
        policy_id,
        "problem.execution_policy.id",
    )


def coerce_execution_policy_config(
    payload: object,
) -> ExecutionPolicyConfig:
    from .strict_deadline_miss import StrictDeadlineMissConfig

    if isinstance(payload, StrictDeadlineMissConfig):
        return payload
    raw_payload = owner_payload(
        payload,
        owner="problem.execution_policy",
        config_type=ExecutionPolicyConfig,
    )
    raw_config = validate_owner_config(raw_payload, ExecutionPolicyConfig)
    _require_execution_policy_id(raw_config.id)
    return validate_owner_config(raw_payload, StrictDeadlineMissConfig)


def compile_execution_policy_contract(
    config: ExecutionPolicyConfig,
) -> CompiledExecutionPolicyContract:
    from .strict_deadline_miss import StrictDeadlineMissConfig, compile_execution_policy

    _require_execution_policy_id(config.id)
    if not isinstance(config, StrictDeadlineMissConfig):
        config = validate_owner_config(config.model_dump(mode="json"), StrictDeadlineMissConfig)
    return compile_execution_policy(config)

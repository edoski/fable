"""Problem-owned realization policy seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from ...core.validation import validate_path_segment
from ...modeling.families.base import ConfigModel
from ...semantics import RealizationPolicySemantics
from ..problem_store import CompiledProblemStore

IntVector = NDArray[np.int64]
BoolMatrix = NDArray[np.bool_]
BoolVector = NDArray[np.bool_]
FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]

class RealizationPolicyConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="problem.realization_policy.id")


@dataclass(frozen=True, slots=True)
class PreparedSupervisedRealizationTargets:
    candidate_mask: BoolMatrix
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
    PreparedSupervisedRealizationTargets,
]
RealizeSelectionsFn = Callable[
    [CompiledProblemStore, DecodedOffsetBatch, IntVector, IntVector],
    RealizedSelectionBatch,
]


@dataclass(frozen=True, slots=True)
class CompiledRealizationPolicyContract:
    realization_policy_id: str
    requires_post_window_row: bool
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


def coerce_realization_policy_config(
    payload: Mapping[str, object] | RealizationPolicyConfig,
) -> RealizationPolicyConfig:
    from .strict_deadline_miss import StrictDeadlineMissConfig

    if isinstance(payload, RealizationPolicyConfig):
        raw_payload = payload.model_dump(mode="json")
        policy_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        policy_id = raw_payload.get("id")
    else:
        raise TypeError("problem.realization_policy must be a mapping or config model")
    if policy_id != "strict_deadline_miss":
        raise ValueError("problem.realization_policy.id must be strict_deadline_miss")
    return StrictDeadlineMissConfig.model_validate(raw_payload)


def compile_realization_policy_contract(
    config: RealizationPolicyConfig,
) -> CompiledRealizationPolicyContract:
    from .strict_deadline_miss import StrictDeadlineMissConfig, compile_realization_policy

    return compile_realization_policy(StrictDeadlineMissConfig.model_validate(config))

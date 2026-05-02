"""Compiled problem contracts shared across workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.models import ChainRuntimeSpec, ProblemSpec
from ..features import CompiledFeatureContract, FeaturePrerequisites
from ..semantics import ProblemSemantics
from .capability import TemporalCapability
from .execution_policy import CompiledExecutionPolicyContract, compile_execution_policy_contract

if TYPE_CHECKING:
    from ..features import ResolvedFeatureTable
    from .problem_store import CompiledProblemStore


@dataclass(frozen=True, slots=True)
class TemporalCapabilityStore:
    store: CompiledProblemStore
    capability: TemporalCapability


@dataclass(frozen=True, slots=True)
class CompiledProblemContract:
    compiler_id: str
    problem_id: str
    features_id: str
    lookback_seconds: int
    sample_count: int
    max_delay_seconds: int
    feature_prerequisites: FeaturePrerequisites
    execution_policy: CompiledExecutionPolicyContract

    @property
    def semantics(self) -> ProblemSemantics:
        return ProblemSemantics(
            compiler_id=self.compiler_id,
            problem_id=self.problem_id,
            lookback_seconds=self.lookback_seconds,
            sample_count=self.sample_count,
            max_delay_seconds=self.max_delay_seconds,
        )

    @property
    def required_history_seconds(self) -> int:
        return self.lookback_seconds + self.feature_prerequisites.history_seconds

    @property
    def warmup_rows(self) -> int:
        return self.feature_prerequisites.warmup_rows

    def initial_history_window_seconds(self, recent_block_interval_seconds: float | None) -> int:
        raise NotImplementedError

    def count_valid_capability_samples(self, feature_table: ResolvedFeatureTable) -> int:
        raise NotImplementedError

    def build_capability_store(
        self,
        feature_table: ResolvedFeatureTable,
    ) -> TemporalCapabilityStore:
        raise NotImplementedError

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        capability: TemporalCapability,
    ) -> CompiledProblemStore:
        raise NotImplementedError


def compile_problem_contract(
    *,
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    chain_runtime: ChainRuntimeSpec | None = None,
) -> CompiledProblemContract:
    from .compilers import compile_problem

    execution_policy = compile_execution_policy_contract(problem.execution_policy)
    return compile_problem(
        problem,
        feature_contract,
        execution_policy,
        chain_runtime,
    )


def problem_runtime_metadata_payload(
    compiler_id: str,
    metadata: object,
) -> dict[str, object]:
    from .compilers import problem_runtime_metadata_payload as compiler_metadata_payload

    return compiler_metadata_payload(compiler_id, metadata)


def problem_runtime_metadata_from_compiler_payload(
    compiler_id: str,
    payload: Mapping[str, object],
) -> object:
    from .compilers import (
        problem_runtime_metadata_from_compiler_payload as compiler_metadata_from_payload,
    )

    return compiler_metadata_from_payload(compiler_id, payload)


def temporal_capability_payload(capability: TemporalCapability) -> dict[str, object]:
    return {
        "compiler_id": capability.compiler_id,
        "max_delay_seconds": capability.max_delay_seconds,
        "action_width": capability.action_width,
        "compiler_runtime_metadata": problem_runtime_metadata_payload(
            capability.compiler_id,
            capability.compiler_runtime_metadata,
        ),
    }


def temporal_capability_from_payload(payload: Mapping[str, object]) -> TemporalCapability:
    compiler_id = _string_payload(payload, "compiler_id")
    return TemporalCapability(
        compiler_id=compiler_id,
        max_delay_seconds=_int_payload(payload, "max_delay_seconds"),
        action_width=_int_payload(payload, "action_width"),
        compiler_runtime_metadata=problem_runtime_metadata_from_compiler_payload(
            compiler_id,
            _mapping_payload(payload, "compiler_runtime_metadata"),
        ),
    )


def _mapping_payload(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"temporal_capability.{key} must be a mapping")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _string_payload(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"temporal_capability.{key} must be a string")
    return value


def _int_payload(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"temporal_capability.{key} must be an integer")
    return value

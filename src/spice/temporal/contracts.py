"""Compiled problem contracts shared across workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.models import ChainRuntimeSpec, ProblemSpec
from ..features import CompiledFeatureContract, FeaturePrerequisites
from ..semantics import ProblemSemantics
from .realization import CompiledRealizationPolicyContract, compile_realization_policy_contract

if TYPE_CHECKING:
    from ..features import ResolvedFeatureTable
    from .problem_store import CompiledProblemStore


@dataclass(frozen=True, slots=True)
class CompiledProblemContract:
    compiler_id: str
    problem_id: str
    feature_set_id: str
    feature_family_id: str
    lookback_seconds: int
    sample_count: int
    max_delay_seconds: int
    feature_prerequisites: FeaturePrerequisites
    realization_policy: CompiledRealizationPolicyContract

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
    ) -> tuple[CompiledProblemStore, object]:
        raise NotImplementedError

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: object,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        raise NotImplementedError


def compile_problem_contract(
    *,
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    chain_runtime: ChainRuntimeSpec | None = None,
) -> CompiledProblemContract:
    from .compilers import compile_problem

    realization_policy = compile_realization_policy_contract(problem.realization_policy)
    return compile_problem(
        problem,
        feature_contract,
        realization_policy,
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

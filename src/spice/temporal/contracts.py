"""Compiled problem contracts shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.models import ProblemSpec
from ..features import CompiledFeatureContract, FeaturePrerequisites

ProblemRuntimeMetadata = dict[str, object]

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
    max_supported_delay_seconds: int
    feature_prerequisites: FeaturePrerequisites

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
    ) -> tuple[CompiledProblemStore, ProblemRuntimeMetadata]:
        raise NotImplementedError

    def build_requested_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        requested_delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        raise NotImplementedError


def compile_problem_contract(
    *,
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
) -> CompiledProblemContract:
    from .compilers.registry import problem_compiler_spec

    return problem_compiler_spec(problem.compiler.id).compile_problem(
        problem,
        feature_contract,
    )

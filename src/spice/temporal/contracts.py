"""Compiled problem contracts shared across workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ..config.models import ChainRuntimeSpec, ProblemSpec
from ..core.errors import ConfigResolutionError
from ..features import CompiledFeatureContract, FeaturePrerequisites
from ..semantics import ProblemSemantics
from .realization import CompiledRealizationPolicyContract, compile_realization_policy_contract

if TYPE_CHECKING:
    from ..features import ResolvedFeatureTable
    from .problem_store import CompiledProblemStore


@dataclass(frozen=True, slots=True)
class TimestampRuntimeMetadata:
    pass


@dataclass(frozen=True, slots=True)
class EstimatedBlockRuntimeMetadata:
    calibrated_interval_seconds: float
    lookback_interval_seconds: float
    candidate_interval_seconds: float
    lookback_steps: int
    capability_candidate_count: int


@dataclass(frozen=True, slots=True)
class TimestampFutureWindowRuntimeMetadata:
    calibrated_interval_seconds: float
    action_interval_seconds: float
    capability_action_count: int


ProblemRuntimeMetadata = (
    TimestampRuntimeMetadata
    | EstimatedBlockRuntimeMetadata
    | TimestampFutureWindowRuntimeMetadata
)


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
    ) -> tuple[CompiledProblemStore, ProblemRuntimeMetadata]:
        raise NotImplementedError

    def build_delay_store(
        self,
        feature_table: ResolvedFeatureTable,
        delay_seconds: int,
        *,
        compiler_runtime_metadata: ProblemRuntimeMetadata,
        max_candidate_slots: int,
    ) -> CompiledProblemStore:
        raise NotImplementedError


def compile_problem_contract(
    *,
    problem: ProblemSpec,
    feature_contract: CompiledFeatureContract,
    chain_runtime: ChainRuntimeSpec | None = None,
) -> CompiledProblemContract:
    from .compilers import problem_compiler_spec

    realization_policy = compile_realization_policy_contract(problem.realization_policy)
    return problem_compiler_spec(problem.compiler.id).compile_problem(
        problem,
        feature_contract,
        realization_policy,
        chain_runtime,
    )


def problem_runtime_metadata_payload(metadata: ProblemRuntimeMetadata) -> dict[str, object]:
    return asdict(metadata)


def problem_runtime_metadata_from_payload(
    problem: ProblemSpec,
    payload: Mapping[str, object],
) -> ProblemRuntimeMetadata:
    return problem_runtime_metadata_from_compiler_payload(problem.compiler.id, payload)


def problem_runtime_metadata_from_compiler_payload(
    compiler_id: str,
    payload: Mapping[str, object],
) -> ProblemRuntimeMetadata:
    raw_payload = dict(payload)
    if compiler_id == "timestamp_native":
        if raw_payload:
            raise ConfigResolutionError(
                "timestamp_native runtime metadata must be empty in artifact manifests"
            )
        return TimestampRuntimeMetadata()
    if compiler_id == "estimated_block":
        return EstimatedBlockRuntimeMetadata(
            calibrated_interval_seconds=_float_payload(
                raw_payload,
                "calibrated_interval_seconds",
            ),
            lookback_interval_seconds=_float_payload(raw_payload, "lookback_interval_seconds"),
            candidate_interval_seconds=_float_payload(raw_payload, "candidate_interval_seconds"),
            lookback_steps=_int_payload(raw_payload, "lookback_steps"),
            capability_candidate_count=_int_payload(raw_payload, "capability_candidate_count"),
        )
    if compiler_id == "timestamp_future_window":
        return TimestampFutureWindowRuntimeMetadata(
            calibrated_interval_seconds=_float_payload(
                raw_payload,
                "calibrated_interval_seconds",
            ),
            action_interval_seconds=_float_payload(raw_payload, "action_interval_seconds"),
            capability_action_count=_int_payload(raw_payload, "capability_action_count"),
        )
    raise ConfigResolutionError(f"Unsupported problem.compiler.id: {compiler_id}")


def _float_payload(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigResolutionError(f"Invalid float runtime metadata field: {key}")
    return float(value)


def _int_payload(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigResolutionError(f"Invalid integer runtime metadata field: {key}")
    return int(value)

"""Corpus capability planning and history sizing policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..acquisition import BlockPullPlan, BlockSource, TimestampRange, evaluation_range
from ..config.models import ChainRuntimeSpec, FeaturesConfig, ProblemSpec
from ..features import CompiledFeatureContract, compile_feature_contract
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract
from .io import load_block_frame
from .metadata import CorpusAcquisitionSourceRequirements
from .validation import BlockDatasetValidationReport

HISTORY_WINDOW_CUSHION_RATIO = 0.10
HISTORY_REFILL_CUSHION_RATIO = 0.10
HISTORY_REFILL_ATTEMPT_LIMIT = 3
CORE_CORPUS_SOURCE_COLUMNS = frozenset({"block_number", "timestamp", "chain_id"})


@dataclass(frozen=True, slots=True)
class CorpusCapabilityPlanningSpec:
    features: FeaturesConfig
    problem: ProblemSpec
    chain_runtime: ChainRuntimeSpec
    history_window_end_timestamp: int
    evaluation_window_start_timestamp: int
    evaluation_window_end_timestamp: int


@dataclass(frozen=True, slots=True)
class InitialCorpusCapabilityPlan:
    history_plan: BlockPullPlan
    evaluation_plan: BlockPullPlan
    requested_history_window_seconds: int


@dataclass(frozen=True, slots=True)
class CorpusHistoryRefillPlan:
    history_plan: BlockPullPlan
    requested_history_window_seconds: int
    status_message: str


@dataclass(frozen=True, slots=True)
class CorpusCapabilityPlanningContext:
    spec: CorpusCapabilityPlanningSpec
    feature_contract: CompiledFeatureContract
    problem_contract: CompiledProblemContract

    @property
    def required_sample_count(self) -> int:
        return self.problem_contract.sample_count

    @property
    def source_requirements(self) -> CorpusAcquisitionSourceRequirements:
        required_columns = frozenset(self.feature_contract.required_source_columns)
        return CorpusAcquisitionSourceRequirements(
            required_columns=CORE_CORPUS_SOURCE_COLUMNS | required_columns,
            optional_enrichments=self.feature_contract.optional_source_enrichments,
            temporal_unit="block",
            ordering_key="block_number",
            partition_key="chain_id",
        )

    async def initial_plan(self, block_source: BlockSource) -> InitialCorpusCapabilityPlan:
        evaluation_window = evaluation_range(
            self.spec.evaluation_window_start_timestamp,
            self.spec.evaluation_window_end_timestamp,
        )
        evaluation_plan = await block_source.plan_window(evaluation_window)
        recent_block_interval_seconds = await block_source.estimate_recent_block_interval()
        bootstrap_history_window_seconds = self.problem_contract.initial_history_window_seconds(
            recent_block_interval_seconds,
        )
        requested_history_window_seconds = _with_cushion(
            bootstrap_history_window_seconds,
            HISTORY_WINDOW_CUSHION_RATIO,
        )
        history_plan = await block_source.plan_window(
            self.history_window(requested_history_window_seconds),
        )
        return InitialCorpusCapabilityPlan(
            history_plan=history_plan,
            evaluation_plan=evaluation_plan,
            requested_history_window_seconds=requested_history_window_seconds,
        )

    def history_window(self, window_seconds: int) -> TimestampRange:
        return TimestampRange(
            start=max(0, self.spec.history_window_end_timestamp - window_seconds),
            end=self.spec.history_window_end_timestamp,
        )

    def count_valid_history_samples(self, history_dir: Path) -> int:
        blocks = load_block_frame(history_dir).sort("block_number")
        feature_table = self.feature_contract.build_table(blocks)
        return self.problem_contract.count_valid_capability_samples(feature_table)

    async def plan_history_refill(
        self,
        *,
        block_source: BlockSource,
        validation: BlockDatasetValidationReport,
        resolved_capability_samples: int,
        requested_history_window_seconds: int,
    ) -> CorpusHistoryRefillPlan | None:
        if resolved_capability_samples >= self.required_sample_count:
            return None
        if (
            validation.first_timestamp is None
            or validation.last_timestamp is None
            or validation.row_count <= 1
        ):
            raise RuntimeError("Cannot compute observed history cadence from validation report")
        sample_shortfall = self.required_sample_count - resolved_capability_samples
        observed_seconds_per_block = max(
            1.0,
            (validation.last_timestamp - validation.first_timestamp)
            / (validation.row_count - 1),
        )
        next_requested_history_window_seconds = max(
            requested_history_window_seconds,
            self.spec.history_window_end_timestamp - validation.first_timestamp,
        ) + _with_cushion(
            sample_shortfall * observed_seconds_per_block,
            HISTORY_REFILL_CUSHION_RATIO,
        )
        if next_requested_history_window_seconds <= requested_history_window_seconds:
            raise RuntimeError(
                "History sizing policy stopped expanding before capability samples were met: "
                f"valid={resolved_capability_samples}, required={self.required_sample_count}"
            )
        history_plan = await block_source.plan_window(
            self.history_window(next_requested_history_window_seconds),
        )
        return CorpusHistoryRefillPlan(
            history_plan=history_plan,
            requested_history_window_seconds=next_requested_history_window_seconds,
            status_message=(
                "history refilling "
                f"samples={resolved_capability_samples}/{self.required_sample_count}"
            ),
        )

    def ensure_sufficient_history_samples(self, resolved_capability_samples: int) -> None:
        if resolved_capability_samples < self.required_sample_count:
            raise RuntimeError(
                "History sizing policy under-requested capability samples: "
                f"valid={resolved_capability_samples}, "
                f"required={self.required_sample_count}, "
                f"refill_attempts={HISTORY_REFILL_ATTEMPT_LIMIT}"
            )


def build_corpus_capability_planning_context(
    spec: CorpusCapabilityPlanningSpec,
) -> CorpusCapabilityPlanningContext:
    feature_contract = compile_feature_contract(features=spec.features)
    problem_contract = compile_problem_contract(
        problem=spec.problem,
        feature_contract=feature_contract,
        chain_runtime=spec.chain_runtime,
    )
    return CorpusCapabilityPlanningContext(
        spec=spec,
        feature_contract=feature_contract,
        problem_contract=problem_contract,
    )


def _with_cushion(value: float, ratio: float) -> int:
    return max(1, math.ceil(value * (1.0 + ratio)))

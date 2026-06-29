"""Artifact-to-inference preparation context."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import polars as pl

from ..config.models import BlockWindowSpec, EvaluateConfig, TimestampWindowSpec
from ..core.errors import ConfigResolutionError, StateConflictError
from ..corpus.coverage import evaluation_coverage_requirement, validate_corpus_coverage
from ..corpus.io import load_block_frame
from ..evaluation import CompiledEvaluatorContract, compile_evaluator_contract
from ..evaluation.contracts import EvaluationSummary
from ..features import compile_feature_contract
from ..prediction import compile_prediction_contract
from ..storage.workflow_roots import ArtifactRootHandle, CorpusRootHandle
from ..temporal.contracts import compile_problem_contract
from .artifacts import load_training_artifact
from .dataset_builders import (
    CompiledInferenceDatasetPreparationRequest,
    EvaluationCoverageWindow,
    PreparedInferenceDataset,
    SampleBlockWindow,
    SampleTimestampWindow,
    prepare_inference_dataset,
)
from .results import (
    EvaluationExecutionProvenance,
    EvaluationRuntimeSummary,
    build_evaluation_runtime_summary,
)
from .runtime_planning import build_cuda_modeling_runtime_plan
from .scoring import EvaluationScoringRuntimePlan


@dataclass(slots=True)
class ArtifactInferenceContext:
    prepared: PreparedInferenceDataset
    delay_seconds: int
    scenario_window_start_timestamp: int
    scenario_window_end_timestamp: int
    required_coverage_start_timestamp: int
    required_coverage_end_timestamp: int
    evaluator_contract: CompiledEvaluatorContract
    scoring_plan: EvaluationScoringRuntimePlan

    def runtime_summary(
        self,
        evaluation: EvaluationSummary,
        *,
        execution_provenance: EvaluationExecutionProvenance | None = None,
    ) -> EvaluationRuntimeSummary:
        return build_evaluation_runtime_summary(
            prepared=self.prepared,
            evaluation=evaluation,
            delay_seconds=self.delay_seconds,
            evaluator_id=self.evaluator_contract.evaluator_id,
            evaluation_config=self.evaluator_contract.config,
            metric_descriptors=self.evaluator_contract.metric_descriptors,
            scenario_window_start_timestamp=self.scenario_window_start_timestamp,
            scenario_window_end_timestamp=self.scenario_window_end_timestamp,
            required_coverage_start_timestamp=self.required_coverage_start_timestamp,
            required_coverage_end_timestamp=self.required_coverage_end_timestamp,
            execution_provenance=execution_provenance,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedEvaluationScenario:
    scenario_start_timestamp: int
    scenario_end_timestamp: int
    sample_timestamp_window: SampleTimestampWindow | None
    sample_block_window: SampleBlockWindow | None


def prepare_artifact_inference_context(
    config: EvaluateConfig,
    *,
    corpus: CorpusRootHandle,
    artifact: ArtifactRootHandle,
) -> ArtifactInferenceContext:
    loaded_artifact = load_training_artifact(artifact.root_path)
    manifest = loaded_artifact.manifest
    corpus_manifest = corpus.load_manifest()
    if manifest.chain_name != corpus_manifest.chain.name:
        raise ConfigResolutionError(
            "evaluation corpus chain does not match artifact chain: "
            f"{corpus_manifest.chain.name} != {manifest.chain_name}"
        )
    feature_contract = compile_feature_contract(features=manifest.features)
    if feature_contract.feature_graph_fingerprint != manifest.feature_graph_fingerprint:
        raise ConfigResolutionError(
            "Current feature graph does not match the trained artifact manifest"
        )
    if feature_contract.feature_prerequisites != manifest.feature_prerequisites:
        raise ConfigResolutionError(
            "Current feature prerequisites do not match the trained artifact manifest"
        )
    problem_contract = compile_problem_contract(
        problem=manifest.problem,
        feature_contract=feature_contract,
        chain_runtime=corpus_manifest.chain.runtime,
    )
    prediction_contract = compile_prediction_contract(
        prediction_id=manifest.prediction.id,
        family_id=manifest.prediction.family_id,
    )
    evaluator_contract = compile_evaluator_contract(config.evaluator)
    capability_max_delay_seconds = manifest.temporal_capability.max_delay_seconds
    delay_seconds = config.delay_seconds or capability_max_delay_seconds
    if delay_seconds > capability_max_delay_seconds:
        raise ConfigResolutionError(
            "delay_seconds exceeds artifact capability: "
            f"{delay_seconds} > {capability_max_delay_seconds}"
        )
    blocks = load_block_frame(corpus.blocks_dir)
    scenario = _resolve_evaluation_scenario(config.evaluation_window, blocks)
    scenario_start = scenario.scenario_start_timestamp
    scenario_end = scenario.scenario_end_timestamp
    training_cutoff = (
        manifest.training_source.training_cutoff_timestamp
        or manifest.training_source.window_end_timestamp
    )
    if scenario_start < training_cutoff:
        raise StateConflictError(
            "evaluation window must be after the artifact training cutoff: "
            f"evaluation_start={scenario_start} "
            f"training_cutoff={training_cutoff}"
        )

    validate_corpus_coverage(
        corpus_manifest,
        contract=problem_contract,
        feature_contract=feature_contract,
        requirement=evaluation_coverage_requirement(
            problem_contract,
            delay_seconds=delay_seconds,
        ),
    )
    required_start, required_end = _required_coverage_window(
        scenario_start=scenario_start,
        scenario_end=scenario_end,
        lookback_seconds=problem_contract.required_history_seconds,
        warmup_rows=problem_contract.warmup_rows,
        delay_seconds=delay_seconds,
        block_spacing_seconds=corpus_manifest.chain.runtime.nominal_block_time_seconds,
    )
    _require_corpus_covers_window(
        corpus_manifest,
        required_start_timestamp=required_start,
        required_end_timestamp=required_end,
    )
    prepared = prepare_inference_dataset(
        blocks.head(0),
        blocks,
        CompiledInferenceDatasetPreparationRequest(
            feature_contract=feature_contract,
            problem_contract=problem_contract,
            delay_seconds=delay_seconds,
            sequence_runtime_metadata=manifest.sequence_runtime_metadata,
            scaler=manifest.scaler,
            temporal_capability=manifest.temporal_capability,
            sample_timestamp_window=scenario.sample_timestamp_window,
            sample_block_window=scenario.sample_block_window,
        ),
    )
    runtime_plan = build_cuda_modeling_runtime_plan(
        batch_size=config.batch_size,
        deterministic=manifest.training.deterministic,
        seed=manifest.training.seed,
    )
    scoring_plan = EvaluationScoringRuntimePlan(
        model=loaded_artifact.model,
        prediction_contract=prediction_contract,
        execution_policy=prepared.execution_policy,
        store=prepared.store,
        action_space=prepared.samples.action_space,
        runtime_plan=runtime_plan,
    )
    return ArtifactInferenceContext(
        prepared=prepared,
        delay_seconds=delay_seconds,
        scenario_window_start_timestamp=scenario_start,
        scenario_window_end_timestamp=scenario_end,
        required_coverage_start_timestamp=required_start,
        required_coverage_end_timestamp=required_end,
        evaluator_contract=evaluator_contract,
        scoring_plan=scoring_plan,
    )


def _resolve_evaluation_scenario(
    window: TimestampWindowSpec | BlockWindowSpec,
    blocks: pl.DataFrame,
) -> _ResolvedEvaluationScenario:
    if isinstance(window, TimestampWindowSpec):
        coverage = EvaluationCoverageWindow(
            first_timestamp=window.start_timestamp,
            last_timestamp=window.end_timestamp - 1,
        )
        return _ResolvedEvaluationScenario(
            scenario_start_timestamp=window.start_timestamp,
            scenario_end_timestamp=window.end_timestamp,
            sample_timestamp_window=coverage.to_sample_timestamp_window(),
            sample_block_window=None,
        )
    selected = (
        blocks.filter(
            (pl.col("block_number") >= window.start_block)
            & (pl.col("block_number") < window.end_block_exclusive)
        )
        .sort("block_number")
    )
    if selected.height != window.block_count:
        raise StateConflictError(
            "evaluation block window is incomplete: "
            f"start_block={window.start_block} "
            f"block_count={window.block_count} found={selected.height}"
        )
    block_numbers = selected["block_number"].cast(pl.Int64).to_numpy()
    if (
        int(block_numbers[0]) != window.start_block
        or int(block_numbers[-1]) != window.end_block_exclusive - 1
        or np.any(np.diff(block_numbers.astype(np.int64, copy=False)) != 1)
    ):
        raise StateConflictError(
            "evaluation block window must contain contiguous block numbers"
        )
    timestamps = selected["timestamp"].cast(pl.Int64).to_numpy()
    return _ResolvedEvaluationScenario(
        scenario_start_timestamp=int(timestamps[0]),
        scenario_end_timestamp=int(timestamps[-1]) + 1,
        sample_timestamp_window=None,
        sample_block_window=SampleBlockWindow(
            start_block_inclusive=window.start_block,
            end_block_exclusive=window.end_block_exclusive,
        ),
    )


def _required_coverage_window(
    *,
    scenario_start: int,
    scenario_end: int,
    lookback_seconds: int,
    warmup_rows: int,
    delay_seconds: int,
    block_spacing_seconds: float,
) -> tuple[int, int]:
    spacing = max(1, ceil(block_spacing_seconds))
    start_support = lookback_seconds + warmup_rows * spacing
    end_support = delay_seconds + spacing
    return scenario_start - start_support, scenario_end + end_support


def _require_corpus_covers_window(
    corpus_manifest,
    *,
    required_start_timestamp: int,
    required_end_timestamp: int,
) -> None:
    coverage = corpus_manifest.blocks.coverage
    covered_start = coverage.first_timestamp
    covered_end = None if coverage.last_timestamp is None else coverage.last_timestamp + 1
    if covered_start is None or covered_end is None:
        raise StateConflictError("Evaluation corpus is missing timestamp coverage")
    if covered_start <= required_start_timestamp and covered_end >= required_end_timestamp:
        return
    missing_start = (
        f"{required_start_timestamp} -> {min(covered_start, required_end_timestamp)}"
        if covered_start > required_start_timestamp
        else None
    )
    missing_end = (
        f"{max(covered_end, required_start_timestamp)} -> {required_end_timestamp}"
        if covered_end < required_end_timestamp
        else None
    )
    missing = ", ".join(part for part in (missing_start, missing_end) if part is not None)
    raise StateConflictError(
        "evaluation corpus does not cover required support window: "
        f"required={required_start_timestamp}->{required_end_timestamp} "
        f"covered={covered_start}->{covered_end} "
        f"missing={missing}; suggested acquire window "
        f"start={required_start_timestamp} end={required_end_timestamp}"
    )

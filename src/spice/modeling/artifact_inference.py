"""Artifact-to-inference preparation context."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import EvaluateConfig
from ..core.errors import ConfigResolutionError
from ..corpus.coverage import evaluation_coverage_requirement, validate_corpus_coverage
from ..corpus.io import load_block_frame
from ..evaluation import CompiledEvaluatorContract, compile_evaluator_contract
from ..features import compile_feature_contract
from ..prediction import compile_prediction_contract
from ..storage.workflow_roots import ArtifactRootHandle, CorpusRootHandle
from ..temporal.contracts import compile_problem_contract
from .artifacts import LoadedTrainingArtifact, load_training_artifact
from .dataset_builders import ArtifactInferencePreparationSpec, PreparedInferenceDataset
from .scoring import ModelScoringInput, build_model_scoring_input


@dataclass(slots=True)
class ArtifactInferenceContext:
    loaded_artifact: LoadedTrainingArtifact
    prepared: PreparedInferenceDataset
    delay_seconds: int
    evaluator_contract: CompiledEvaluatorContract
    scoring_input: ModelScoringInput


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
    evaluator_contract = compile_evaluator_contract(config.evaluation)
    delay_seconds = config.delay_seconds or manifest.max_delay_seconds
    if delay_seconds > manifest.max_delay_seconds:
        raise ConfigResolutionError(
            "delay_seconds exceeds artifact capability: "
            f"{delay_seconds} > {manifest.max_delay_seconds}"
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
    prepared = loaded_artifact.dataset_builder_contract.prepare_inference_dataset(
        load_block_frame(corpus.history_dir),
        load_block_frame(corpus.evaluation_dir),
        spec=ArtifactInferencePreparationSpec(
            feature_contract=feature_contract,
            problem_contract=problem_contract,
            delay_seconds=delay_seconds,
            builder_runtime_metadata=manifest.builder_runtime_metadata,
            scaler=manifest.scaler,
            max_candidate_slots=manifest.max_candidate_slots,
            evaluation_start_timestamp=corpus_manifest.coverage.evaluation.start_timestamp,
            evaluation_end_timestamp=corpus_manifest.coverage.evaluation.end_timestamp,
        ),
    )
    scoring_input = build_model_scoring_input(
        model=loaded_artifact.model,
        model_config=loaded_artifact.manifest.model,
        prediction_contract=prediction_contract,
        representation_contract=loaded_artifact.representation_contract,
        execution_policy=prepared.execution_policy,
        store=prepared.store,
        sample_indices=prepared.sample_indices,
        batch_size=config.batch_size,
    )
    return ArtifactInferenceContext(
        loaded_artifact=loaded_artifact,
        prepared=prepared,
        delay_seconds=delay_seconds,
        evaluator_contract=evaluator_contract,
        scoring_input=scoring_input,
    )

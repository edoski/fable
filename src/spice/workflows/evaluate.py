"""Evaluation workflow."""

from __future__ import annotations

from ..config.models import EvaluateConfig
from ..core.errors import ConfigResolutionError
from ..core.reporting import Reporter
from ..corpus.io import load_block_frame
from ..evaluation import compile_evaluator_contract
from ..modeling.artifacts import load_training_artifact, validate_artifact_semantics
from ..modeling.evaluation import run_prediction_evaluation
from ..modeling.inference import predict_with_model
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.results import LoadedEvaluationSummary, build_evaluation_runtime_summary
from ..modeling.summary import evaluation_result_fields
from ..modeling.tuning import apply_study_best_params
from ..prediction import compile_prediction_contract
from ..storage.artifact import write_evaluation_state
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.layout import resolve_workflow_paths
from ..temporal.contracts import compile_problem_contract


def _workflow_facts(config: EvaluateConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("delay", f"{config.delay_seconds}s"),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
        ("evaluation", config.evaluation.id),
    ]
    if config.artifact.variant.value == "tuned":
        facts.append(("study", config.study.name))
    return facts


def run(config: EvaluateConfig, *, reporter: Reporter | None = None) -> None:
    active_config = config
    study_id: str | None = None
    if config.artifact.variant.value == "tuned":
        applied = apply_study_best_params(config)
        active_config = applied.config
        study_id = applied.study_id
    paths = resolve_workflow_paths(active_config, study_id=study_id)
    artifact_dir = paths.artifact_root
    history_block_path = paths.history_dir
    evaluation_block_path = paths.evaluation_dir
    if artifact_dir is None:
        raise ConfigResolutionError("evaluation workflow requires artifact output paths")
    active_reporter = reporter or Reporter()
    active_reporter.header("evaluate", _workflow_facts(active_config))
    loaded_artifact = load_training_artifact(artifact_dir)
    validated = validate_artifact_semantics(
        loaded_artifact.manifest,
        problem=active_config.problem,
        dataset_builder=active_config.dataset_builder,
        feature_set=active_config.feature_set,
        prediction=active_config.prediction,
        objective=active_config.objective,
        model=active_config.model,
        split=active_config.split,
        training=active_config.training,
    )
    feature_contract = validated.feature_contract
    history_blocks = load_block_frame(history_block_path)
    evaluation_blocks = load_block_frame(evaluation_block_path)
    contract = compile_problem_contract(
        problem=active_config.problem,
        feature_contract=feature_contract,
        chain_runtime=active_config.chain.runtime,
    )
    prediction_contract = compile_prediction_contract(
        prediction_id=active_config.prediction.id,
        family_id=active_config.prediction.family_id,
    )
    if active_config.evaluation is None:
        raise ConfigResolutionError("evaluation workflow requires evaluation")
    evaluator_contract = compile_evaluator_contract(active_config.evaluation)
    if active_config.delay_seconds > loaded_artifact.manifest.max_delay_seconds:
        raise ConfigResolutionError(
            "delay_seconds exceeds artifact capability: "
            f"{active_config.delay_seconds} > {loaded_artifact.manifest.max_delay_seconds}"
        )
    prepared = prepare_inference_dataset(
        history_blocks,
        evaluation_blocks,
        dataset_builder_contract=validated.dataset_builder_contract,
        feature_contract=feature_contract,
        contract=contract,
        delay_seconds=active_config.delay_seconds,
        builder_runtime_metadata=loaded_artifact.manifest.builder_runtime_metadata,
        scaler=loaded_artifact.manifest.scaler,
        max_candidate_slots=loaded_artifact.manifest.max_candidate_slots,
        window_start_timestamp=active_config.evaluation_window_start_timestamp,
        window_end_timestamp=active_config.evaluation_window_end_timestamp,
    )
    active_reporter.milestone(
        "prepare "
        f"history_rows={prepared.n_history_rows} "
        f"evaluation_rows={prepared.n_evaluation_rows} "
        f"samples={prepared.sample_count}"
    )
    decoded_offsets = predict_with_model(
        loaded_artifact.model,
        model_config=loaded_artifact.manifest.model,
        prediction_contract=prediction_contract,
        representation_contract=loaded_artifact.representation_contract,
        store=prepared.store,
        sample_indices=prepared.sample_indices,
        batch_size=active_config.training.batch_size,
    )
    evaluation = run_prediction_evaluation(
        evaluator_contract,
        prepared.store,
        prepared.realization_policy,
        decoded_offsets,
        sample_indices=prepared.sample_indices,
    )
    runtime_summary = build_evaluation_runtime_summary(
        prepared=prepared,
        evaluation=evaluation,
        delay_seconds=active_config.delay_seconds,
        evaluation_id=evaluator_contract.evaluation_id,
        evaluation_config=evaluator_contract.config_payload,
        metric_descriptors=evaluator_contract.metric_descriptors,
    )
    evaluation_id, recorded_at = write_evaluation_state(
        artifact_dir / ".spice" / "state.sqlite",
        root_kind=ARTIFACT_ROOT_KIND,
        summary=runtime_summary,
    )
    summary = LoadedEvaluationSummary(
        evaluation_id=evaluation_id,
        recorded_at=recorded_at,
        manifest=loaded_artifact.manifest,
        runtime=runtime_summary,
    )
    active_reporter.result("evaluate", evaluation_result_fields(summary))

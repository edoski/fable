"""Evaluation workflow."""

from __future__ import annotations

from ..config import EvaluateConfig
from ..core.errors import ConfigResolutionError
from ..core.reporting import Reporter
from ..corpus.io import load_block_frame
from ..evaluation import compile_evaluator_contract
from ..modeling.artifacts import load_training_artifact, validate_artifact_semantics
from ..modeling.evaluation import run_prediction_evaluation
from ..modeling.inference import predict_with_model
from ..modeling.pipeline import prepare_inference_dataset
from ..modeling.results import LoadedEvaluationSummary, build_evaluation_runtime_summary
from ..modeling.summary import evaluation_summary_sections
from ..modeling.tuning import apply_study_best_params
from ..prediction import compile_prediction_contract
from ..storage.artifact import write_evaluation_state
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.layout import resolve_workflow_paths
from ..temporal.contracts import compile_problem_contract
from ._shared import abort_cleanup, managed_workflow


def _workflow_facts(config: EvaluateConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("delay", f"{config.delay_seconds}s"),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
        ("evaluator", config.evaluation.evaluator.id),
    ]
    if config.artifact.variant.value == "tuned":
        facts.append(("study", config.study.name))
    return facts


def run(config: EvaluateConfig, *, reporter: Reporter | None = None) -> None:
    active_config = config
    if config.artifact.variant.value == "tuned":
        active_config = apply_study_best_params(config)
    paths = resolve_workflow_paths(active_config)
    artifact_dir = paths.artifact_root
    history_block_path = paths.history_dir
    evaluation_block_path = paths.evaluation_dir
    if artifact_dir is None:
        raise ConfigResolutionError("evaluation workflow requires artifact output paths")
    with managed_workflow(reporter=reporter) as session:
        session.runtime.configure_workflow("evaluate", _workflow_facts(active_config))
        load_reporter = session.runtime.stage_reporter("load", label="load")
        prepare_reporter = session.runtime.stage_reporter("prepare", label="prepare")
        predict_reporter = session.runtime.stage_reporter("predict", label="predict")
        evaluation_reporter = session.runtime.stage_reporter("evaluate", label="evaluate")
        write_reporter = session.runtime.stage_reporter(
            "write",
            label="write",
            running_status="writing",
        )
        with abort_cleanup(
            session.reporter,
            label="evaluate",
            cleanup=lambda: None,
        ):
            load_task = load_reporter.start_task("load evaluation inputs")
            loaded_artifact = load_training_artifact(artifact_dir)
            validated = validate_artifact_semantics(
                loaded_artifact.manifest,
                problem=active_config.problem,
                dataset_builder=active_config.dataset_builder,
                feature_set=active_config.feature_set,
                prediction=active_config.prediction,
                objective=active_config.objective,
                model=active_config.model,
            )
            feature_contract = validated.feature_contract
            history_blocks = load_block_frame(history_block_path)
            evaluation_blocks = load_block_frame(evaluation_block_path)
            load_reporter.finish_task(
                load_task,
                message=f"artifact={artifact_dir} evaluation={evaluation_block_path}",
            )
            prepare_task = prepare_reporter.start_task("prepare inference dataset")
            contract = compile_problem_contract(
                problem=active_config.problem,
                feature_contract=feature_contract,
                chain_runtime=active_config.chain.runtime,
            )
            prediction_contract = compile_prediction_contract(
                prediction_id=active_config.prediction.id,
                family_config=active_config.prediction.family,
            )
            evaluator_contract = compile_evaluator_contract(active_config.evaluation.evaluator)
            if "evaluate" not in prediction_contract.supported_workflows:
                raise ConfigResolutionError(
                    f"prediction family {prediction_contract.prediction_family_id} "
                    "does not support evaluate"
                )
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
            prepare_reporter.finish_task(
                prepare_task,
                message=f"samples={prepared.sample_count}",
            )
            decoded_offsets = predict_with_model(
                loaded_artifact.model,
                prediction_contract=prediction_contract,
                representation_contract=loaded_artifact.representation_contract,
                store=prepared.store,
                sample_indices=prepared.sample_indices,
                batch_size=active_config.training.batch_size,
                device=active_config.training.device,
                reporter=predict_reporter,
            )
            evaluation = run_prediction_evaluation(
                evaluator_contract,
                prepared.store,
                prepared.realization_policy,
                decoded_offsets,
                sample_indices=prepared.sample_indices,
                reporter=evaluation_reporter,
            )
            runtime_summary = build_evaluation_runtime_summary(
                prepared=prepared,
                evaluation=evaluation,
                delay_seconds=active_config.delay_seconds,
                evaluator_id=evaluator_contract.evaluator_id,
                evaluator_config=evaluator_contract.config_payload,
                metric_descriptors=evaluator_contract.metric_descriptors,
            )
            report_task = write_reporter.start_task("write evaluation state")
            evaluation_id, recorded_at = write_evaluation_state(
                artifact_dir / ".spice" / "state.sqlite",
                root_kind=ARTIFACT_ROOT_KIND,
                summary=runtime_summary,
            )
            write_reporter.finish_task(report_task, message=str(artifact_dir), silent=True)
            summary = LoadedEvaluationSummary(
                evaluation_id=evaluation_id,
                recorded_at=recorded_at,
                manifest=loaded_artifact.manifest,
                runtime=runtime_summary,
            )
        session.runtime.log_sectioned_summary(
            "evaluation summary",
            evaluation_summary_sections(summary),
        )

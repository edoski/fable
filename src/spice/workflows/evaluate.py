"""Evaluation workflow."""

from __future__ import annotations

import os

from ..config.models import EvaluateConfig
from ..core.reporting import Reporter
from ..modeling.artifact_inference import prepare_artifact_inference_context
from ..modeling.results import (
    EvaluationExecutionProvenance,
    LoadedEvaluationSummary,
    build_evaluation_runtime_summary,
)
from ..modeling.scoring import score_evaluation
from ..modeling.summary import evaluation_result_fields
from ..storage.workflow_roots import EvaluateWorkflowRoots, resolve_evaluate_roots


def _workflow_facts(
    config: EvaluateConfig,
    roots: EvaluateWorkflowRoots,
) -> list[tuple[str, str]]:
    facts = [
        ("dataset", roots.corpus.dataset_name),
        ("dataset_id", roots.corpus.dataset_id),
        ("artifact_id", roots.artifact.artifact_id),
        ("delay", "artifact_max" if config.delay_seconds is None else f"{config.delay_seconds}s"),
        ("evaluation", config.evaluation.id),
        ("batch_size", str(config.batch_size)),
    ]
    return facts


def run(config: EvaluateConfig, *, reporter: Reporter | None = None) -> None:
    roots = resolve_evaluate_roots(config)
    active_reporter = reporter or Reporter()
    active_reporter.header("evaluate", _workflow_facts(config, roots))
    inference_context = prepare_artifact_inference_context(
        config,
        corpus=roots.corpus,
        artifact=roots.artifact,
    )
    prepared = inference_context.prepared
    active_reporter.milestone(
        "prepare "
        f"history_rows={prepared.n_history_rows} "
        f"evaluation_rows={prepared.n_evaluation_rows} "
        f"samples={prepared.sample_count}"
    )
    evaluation = score_evaluation(
        model_input=inference_context.scoring_input,
        evaluator_contract=inference_context.evaluator_contract,
    )
    runtime_summary = build_evaluation_runtime_summary(
        prepared=prepared,
        evaluation=evaluation,
        delay_seconds=inference_context.delay_seconds,
        evaluation_id=inference_context.evaluator_contract.evaluation_id,
        evaluation_config=inference_context.evaluator_contract.config_payload,
        metric_descriptors=inference_context.evaluator_contract.metric_descriptors,
        execution_provenance=_current_execution_provenance(),
    )
    evaluation_id, recorded_at = roots.artifact.upsert_evaluation_state(runtime_summary)
    summary = LoadedEvaluationSummary(
        evaluation_id=evaluation_id,
        recorded_at=recorded_at,
        manifest=inference_context.loaded_artifact.manifest,
        runtime=runtime_summary,
    )
    active_reporter.result("evaluate", evaluation_result_fields(summary))


def _current_execution_provenance() -> EvaluationExecutionProvenance | None:
    execution_ref = os.environ.get("SPICE_EXECUTION_REF")
    if not execution_ref or execution_ref.endswith(":"):
        return None
    return EvaluationExecutionProvenance(
        execution_ref=execution_ref,
        job_id=os.environ.get("SPICE_EXECUTION_JOB_ID") or None,
        log_path=os.environ.get("SPICE_EXECUTION_LOG_PATH") or None,
        workflow_task=os.environ.get("SPICE_WORKFLOW_TASK") or None,
        target=os.environ.get("SPICE_EXECUTION_TARGET") or None,
    )

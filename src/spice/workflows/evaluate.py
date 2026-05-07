"""Evaluation workflow."""

from __future__ import annotations

import os

from ..config.models import EvaluateConfig
from ..core.reporting import Reporter
from ..modeling.results import (
    EvaluationExecutionProvenance,
    build_evaluation_runtime_summary,
)
from ..modeling.scoring import score_evaluation
from ..storage.transactions import record_artifact_evaluation_state
from .preparation import prepare_evaluate
from .reporting import (
    evaluate_workflow_facts,
    report_evaluate_prepare,
    report_evaluate_result,
)


def run(config: EvaluateConfig, *, reporter: Reporter | None = None) -> None:
    prepared_workflow = prepare_evaluate(config)
    roots = prepared_workflow.roots
    active_reporter = reporter or Reporter()
    active_reporter.header("evaluate", evaluate_workflow_facts(config, roots))
    inference_context = prepared_workflow.inference_context
    prepared = inference_context.prepared
    report_evaluate_prepare(
        active_reporter,
        n_history_rows=prepared.n_history_rows,
        n_evaluation_rows=prepared.n_evaluation_rows,
        sample_count=prepared.sample_count,
    )
    evaluation = score_evaluation(
        scoring_plan=inference_context.scoring_plan,
        evaluator_contract=inference_context.evaluator_contract,
    )
    runtime_summary = build_evaluation_runtime_summary(
        prepared=prepared,
        evaluation=evaluation,
        delay_seconds=inference_context.delay_seconds,
        evaluator_id=inference_context.evaluator_contract.evaluator_id,
        evaluation_config=inference_context.evaluator_contract.config,
        metric_descriptors=inference_context.evaluator_contract.metric_descriptors,
        execution_provenance=_current_execution_provenance(),
    )
    summary = record_artifact_evaluation_state(
        roots.artifact,
        summary=runtime_summary,
    )
    report_evaluate_result(active_reporter, summary=summary)


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

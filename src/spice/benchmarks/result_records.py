# pyright: strict

"""Typed benchmark result records stored in collection snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config.models import WorkflowTask
from ..modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary
from .models import LoadedBenchmarkPlanEntry
from .runs import BenchmarkSubmissionRecord, format_datetime


class MetricValueRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    metric_id: str
    value: float


class WindowMetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    mean: float
    std: float


class EvaluationRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, str | int | float]


class BenchmarkResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    step_id: str
    workflow: WorkflowTask
    depends_on: tuple[str, ...]
    external_dependencies: tuple[str, ...]
    dimension_labels: dict[str, str]
    selection: dict[str, object]
    artifact_from: str | None

    job_id: str
    execution_ref: str
    git_commit: str
    dependency: str | None
    log_path: str

    evaluation_execution_ref: str | None
    evaluation_job_id: str | None
    evaluation_log_path: str | None
    evaluation_workflow_task: str | None
    evaluation_target: str | None

    artifact_id: str
    evaluation_storage_id: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    features_id: str
    model_id: str
    problem_id: str
    prediction_id: str
    objective_id: str
    evaluation_id: str
    delay_seconds: int
    variant: str
    study_id: str | None
    study_name: str | None
    recorded_at_utc: str
    sample_count: int
    total_events: int
    n_history_rows: int
    n_evaluation_rows: int

    metrics: tuple[MetricValueRecord, ...]
    window_metrics: tuple[WindowMetricRecord, ...]
    evaluation_runs: tuple[EvaluationRunRecord, ...]


class BenchmarkCollectionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    benchmark: str
    run_dir: str
    target: str
    run_created_at_utc: str
    collected_at_utc: str
    expected_evaluate_count: int
    records: tuple[BenchmarkResultRecord, ...]


def build_benchmark_result_record(
    *,
    entry: LoadedBenchmarkPlanEntry,
    submission: BenchmarkSubmissionRecord,
    evaluation: LoadedEvaluationSummary,
    training: LoadedTrainingSummary | None,
    collector_time: datetime,
) -> BenchmarkResultRecord:
    manifest = evaluation.manifest
    runtime = evaluation.runtime
    metrics: list[MetricValueRecord] = []
    if training is not None:
        metrics.extend(
            MetricValueRecord(source="training_test", metric_id=metric_id, value=value)
            for metric_id, value in training.runtime.test_metrics.values.items()
        )
    metrics.extend(
        MetricValueRecord(source="evaluation", metric_id=metric_id, value=value)
        for metric_id, value in runtime.metrics.values.items()
    )
    recorded_at = (
        format_datetime(datetime.fromtimestamp(evaluation.recorded_at, UTC))
        if evaluation.recorded_at > 0
        else format_datetime(collector_time)
    )
    provenance = runtime.execution_provenance
    return BenchmarkResultRecord(
        run_id=entry.run_id,
        case_id=entry.case_id,
        step_id=entry.step_id,
        workflow=entry.workflow,
        depends_on=entry.depends_on,
        external_dependencies=entry.external_dependencies,
        dimension_labels=dict(entry.dimension_labels),
        selection=dict(entry.selection),
        artifact_from=entry.artifact_from,
        job_id=submission.job_id,
        execution_ref=submission.execution_ref,
        git_commit=submission.git_commit,
        dependency=submission.dependency,
        log_path=submission.log_path,
        evaluation_execution_ref=None if provenance is None else provenance.execution_ref,
        evaluation_job_id=None if provenance is None else provenance.job_id,
        evaluation_log_path=None if provenance is None else provenance.log_path,
        evaluation_workflow_task=None if provenance is None else provenance.workflow_task,
        evaluation_target=None if provenance is None else provenance.target,
        artifact_id=manifest.artifact_id,
        evaluation_storage_id=evaluation.evaluation_id,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        chain_name=manifest.chain_name,
        features_id=manifest.features_id,
        model_id=manifest.model.id,
        problem_id=manifest.problem_id,
        prediction_id=manifest.prediction_id,
        objective_id=manifest.objective.id,
        evaluation_id=runtime.evaluation_id,
        delay_seconds=runtime.delay_seconds,
        variant=manifest.variant.value,
        study_id=manifest.study_id,
        study_name=None if manifest.study is None else manifest.study.name,
        recorded_at_utc=recorded_at,
        sample_count=runtime.sample_count,
        total_events=runtime.total_events,
        n_history_rows=runtime.n_history_rows,
        n_evaluation_rows=runtime.n_evaluation_rows,
        metrics=tuple(metrics),
        window_metrics=tuple(
            WindowMetricRecord(metric_id=metric_id, mean=summary.mean, std=summary.std)
            for metric_id, summary in runtime.window_metrics.items()
        ),
        evaluation_runs=tuple(
            EvaluationRunRecord(
                ordinal=ordinal,
                n_events=run.n_events,
                metrics=dict(run.metrics),
                metadata=_metadata_payload(run.metadata),
            )
            for ordinal, run in enumerate(runtime.runs, start=1)
        ),
    )


def _metadata_payload(values: dict[str, Any]) -> dict[str, str | int | float]:
    return {key: _metadata_value(value) for key, value in values.items()}


def _metadata_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)

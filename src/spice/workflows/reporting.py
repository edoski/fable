"""Workflow runner reporting composition."""

from __future__ import annotations

from ..config.models import EvaluateConfig, TrainConfig, TuneConfig
from ..core.rendering import metric_string
from ..core.reporting import Reporter
from ..modeling._fit_policy import TrainingEpochProgress
from ..modeling.pipeline import TrainingRunCallbacks, TrainingSpec
from ..modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary
from ..modeling.summary import training_result_fields
from ..modeling.tuning_execution import (
    TuningBestProgress,
    TuningExecutionCallbacks,
    TuningTrialProgress,
)
from ..storage.study_models import StudySummary
from ..storage.study_render import study_result_fields
from ..storage.workflow_roots import (
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
)


def train_workflow_facts(
    config: TrainConfig,
    roots: TrainWorkflowRoots,
) -> list[tuple[str, str]]:
    facts = [
        ("corpus", roots.corpus.corpus_name),
        ("corpus_id", roots.corpus.corpus_id),
        ("chain", roots.corpus.chain_name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
        ("artifact_id", roots.artifact.artifact_id),
    ]
    if isinstance(roots, TunedTrainWorkflowRoots):
        facts.append(("study", roots.study.study_name))
        facts.append(("study_id", roots.study.study_id))
    return facts


def report_train_prepare_complete(
    reporter: Reporter,
    *,
    n_rows_used: int,
    sample_count: int,
) -> None:
    reporter.milestone(f"prepare rows={n_rows_used} samples={sample_count}")


def report_train_fit_start(reporter: Reporter, *, max_epochs: int) -> None:
    reporter.milestone(f"fit started epochs={max_epochs}")


def report_train_epoch(
    reporter: Reporter,
    progress: TrainingEpochProgress,
    *,
    primary_metric_id: str,
) -> None:
    reporter.milestone(_fit_epoch_message(progress, primary_metric_id=primary_metric_id))


def report_train_early_stop(
    reporter: Reporter,
    *,
    epoch: int,
    best_epoch: int,
) -> None:
    reporter.milestone(f"fit early_stop epoch={epoch} best_epoch={best_epoch}")


def train_reporting_callbacks(
    reporter: Reporter,
    *,
    spec: TrainingSpec,
) -> TrainingRunCallbacks:
    return TrainingRunCallbacks(
        on_prepare_complete=lambda prepared: report_train_prepare_complete(
            reporter,
            n_rows_used=prepared.n_rows_used,
            sample_count=prepared.sample_count,
        ),
        on_fit_start=lambda: report_train_fit_start(
            reporter,
            max_epochs=spec.training.max_epochs,
        ),
        on_epoch_end=lambda progress: report_train_epoch(
            reporter,
            progress,
            primary_metric_id=spec.prediction_contract.primary_metric_id,
        ),
        on_early_stop=lambda epoch, best_epoch: report_train_early_stop(
            reporter,
            epoch=epoch,
            best_epoch=best_epoch,
        ),
    )


def report_train_result(
    reporter: Reporter,
    *,
    summary: LoadedTrainingSummary,
    artifact_dir,
) -> None:
    reporter.result(
        "train",
        training_result_fields(
            summary,
            artifact_dir=artifact_dir,
        ),
    )


def tune_workflow_facts(config: TuneConfig, roots: TuneWorkflowRoots) -> list[tuple[str, str]]:
    return [
        ("corpus", roots.corpus.corpus_name),
        ("corpus_id", roots.corpus.corpus_id),
        ("chain", roots.corpus.chain_name),
        ("problem", config.problem.id),
        ("features", config.features.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("study", roots.study.study_name),
        ("study_id", roots.study.study_id),
        ("trials", str(config.tuning.trial_count)),
    ]


def report_tune_resume(reporter: Reporter, *, existing: int, target: int) -> None:
    reporter.milestone(f"resume trials={existing}/{target}")


def report_tune_study_start(reporter: Reporter, *, remaining: int) -> None:
    reporter.milestone(f"study started trials={remaining}")


def report_tune_trial(reporter: Reporter, progress: TuningTrialProgress) -> None:
    reporter.milestone(_trial_message(progress))


def report_tune_best(reporter: Reporter, progress: TuningBestProgress) -> None:
    reporter.milestone(
        "best improved "
        f"trial={progress.trial_number + 1} "
        f"value={metric_string(progress.value)}"
    )


def report_tune_result(reporter: Reporter, *, summary: StudySummary) -> None:
    reporter.result("tune", study_result_fields(summary))


def tune_reporting_callbacks(reporter: Reporter) -> TuningExecutionCallbacks:
    return TuningExecutionCallbacks(
        on_resume=lambda existing, target: report_tune_resume(
            reporter,
            existing=existing,
            target=target,
        ),
        on_study_start=lambda remaining: report_tune_study_start(
            reporter,
            remaining=remaining,
        ),
        on_trial_complete=lambda progress: report_tune_trial(
            reporter,
            progress,
        ),
        on_best_improved=lambda progress: report_tune_best(
            reporter,
            progress,
        ),
    )


def evaluate_workflow_facts(
    config: EvaluateConfig,
    roots: EvaluateWorkflowRoots,
) -> list[tuple[str, str]]:
    return [
        ("corpus", roots.corpus.corpus_name),
        ("corpus_id", roots.corpus.corpus_id),
        ("artifact_id", roots.artifact.artifact_id),
        ("delay", "artifact_max" if config.delay_seconds is None else f"{config.delay_seconds}s"),
        ("evaluator", config.evaluator.id),
        ("batch_size", str(config.batch_size)),
    ]


def report_evaluate_prepare(
    reporter: Reporter,
    *,
    n_history_rows: int,
    n_evaluation_rows: int,
    sample_count: int,
) -> None:
    reporter.milestone(
        "prepare "
        f"history_rows={n_history_rows} "
        f"evaluation_rows={n_evaluation_rows} "
        f"samples={sample_count}"
    )


def report_evaluate_result(
    reporter: Reporter,
    *,
    summary: LoadedEvaluationSummary,
) -> None:
    reporter.result("evaluate", _evaluation_result_fields(summary))


def _evaluation_result_fields(summary: LoadedEvaluationSummary) -> list[tuple[str, str]]:
    runtime = summary.runtime
    fields = [
        ("evaluation_storage_id", summary.evaluation_storage_id),
        ("events", str(runtime.total_events)),
    ]
    primary_descriptor = next(
        (descriptor for descriptor in runtime.metric_descriptors if descriptor.role == "primary"),
        None,
    )
    if primary_descriptor is not None and primary_descriptor.id in runtime.metrics.values:
        fields.append(
            (
                primary_descriptor.id,
                metric_string(runtime.metrics.values[primary_descriptor.id]),
            )
        )
    return fields


def _fit_epoch_message(
    progress: TrainingEpochProgress,
    *,
    primary_metric_id: str,
) -> str:
    fields = [f"epoch={progress.epoch}/{progress.max_epochs}"]
    if primary_metric_id in progress.validation_metrics.values:
        fields.append(
            f"validation.{primary_metric_id}="
            f"{metric_string(progress.validation_metrics.values[primary_metric_id])}"
        )
    fields.append(f"best_epoch={progress.best_epoch}")
    fields.append(
        f"best.validation.total_loss={metric_string(progress.best_validation_loss)}"
    )
    return "fit " + " ".join(fields)


def _trial_message(progress: TuningTrialProgress) -> str:
    parts = [f"trial {progress.number + 1}/{progress.total_trials}"]
    if progress.state == "COMPLETE":
        parts.append("complete")
        if progress.value is not None:
            parts.append(f"value={metric_string(progress.value)}")
        if progress.best_epoch is not None:
            parts.append(f"best_epoch={progress.best_epoch}")
        return " ".join(parts)
    if progress.state == "PRUNED":
        parts.append("pruned")
        return " ".join(parts)
    parts.append("failed")
    return " ".join(parts)

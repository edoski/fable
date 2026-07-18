# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from ...config import (
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
)
from ...config.models import TimestampWindowSpec
from ...config.resolution import resolve_workflow_config
from ...config.resolved_workflows import ResolvedWorkflowConfig
from ...core.reporting import Reporter
from ...execution.submission import WorkflowSubmissionEvent, submit_resolved_workflow
from ..options import (
    DEFAULT_REMOTE_TARGET,
    RemoteTargetOption,
    WorkflowArtifactConsumerOption,
    WorkflowBatchSizeOption,
    WorkflowChainOption,
    WorkflowCorpusConsumerOption,
    WorkflowDelaySecondsOption,
    WorkflowDependencyOption,
    WorkflowDetachOption,
    WorkflowEvaluationCorpusOption,
    WorkflowEvaluationWindowDurationOption,
    WorkflowEvaluationWindowStartOption,
    WorkflowEvaluatorSpecOption,
    WorkflowFeaturesOption,
    WorkflowModelOption,
    WorkflowProblemOption,
    WorkflowSplitOption,
    WorkflowStudyConsumerOption,
    WorkflowStudyOption,
    WorkflowSurfaceOption,
    WorkflowTrainingOption,
    WorkflowTrialCountOption,
    WorkflowTuningOption,
    WorkflowTuningSpaceOption,
    WorkflowVariantOption,
)


def _submit_selected_workflow(
    *,
    config: ResolvedWorkflowConfig,
    target: str,
    dependency: str | None,
    detach: bool,
) -> None:
    reporter = Reporter()
    submit_resolved_workflow(
        config,
        target=target,
        dependency=dependency,
        detach=detach,
        on_event=lambda event: _report_workflow_submission_event(reporter, event),
    )


def _submit_model_workflow(
    *,
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
    target: str,
    dependency: str | None,
    detach: bool,
) -> None:
    config = resolve_workflow_config(selection)
    _submit_selected_workflow(
        config=config,
        target=target,
        dependency=dependency,
        detach=detach,
    )


def train_command(
    surface: WorkflowSurfaceOption = None,
    chain: WorkflowChainOption = None,
    problem: WorkflowProblemOption = None,
    features: WorkflowFeaturesOption = None,
    model: WorkflowModelOption = None,
    tuning_space: WorkflowTuningSpaceOption = None,
    training: WorkflowTrainingOption = None,
    split: WorkflowSplitOption = None,
    tuning: WorkflowTuningOption = None,
    study: WorkflowStudyOption = None,
    variant: WorkflowVariantOption = None,
    corpus_id: WorkflowCorpusConsumerOption = None,
    study_id: WorkflowStudyConsumerOption = None,
    dependency: WorkflowDependencyOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: WorkflowDetachOption = False,
) -> None:
    _submit_model_workflow(
        selection=TrainWorkflowSelection(
            surface=surface,
            chain=chain,
            problem=problem,
            features=features,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            corpus_id=corpus_id,
            study_id=study_id,
            variant=variant,
        ),
        target=target,
        dependency=dependency,
        detach=detach,
    )


def tune_command(
    surface: WorkflowSurfaceOption = None,
    chain: WorkflowChainOption = None,
    problem: WorkflowProblemOption = None,
    features: WorkflowFeaturesOption = None,
    model: WorkflowModelOption = None,
    tuning_space: WorkflowTuningSpaceOption = None,
    training: WorkflowTrainingOption = None,
    split: WorkflowSplitOption = None,
    tuning: WorkflowTuningOption = None,
    study: WorkflowStudyOption = None,
    corpus_id: WorkflowCorpusConsumerOption = None,
    trial_count: WorkflowTrialCountOption = None,
    dependency: WorkflowDependencyOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: WorkflowDetachOption = False,
) -> None:
    _submit_model_workflow(
        selection=TuneWorkflowSelection(
            surface=surface,
            chain=chain,
            problem=problem,
            features=features,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            corpus_id=corpus_id,
            trial_count=trial_count,
        ),
        target=target,
        dependency=dependency,
        detach=detach,
    )


def evaluate_command(
    artifact_id: WorkflowArtifactConsumerOption = None,
    corpus_id: WorkflowEvaluationCorpusOption = None,
    evaluator: WorkflowEvaluatorSpecOption = None,
    evaluation_start: WorkflowEvaluationWindowStartOption = None,
    evaluation_duration_seconds: WorkflowEvaluationWindowDurationOption = None,
    delay_seconds: WorkflowDelaySecondsOption = None,
    batch_size: WorkflowBatchSizeOption = None,
    dependency: WorkflowDependencyOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: WorkflowDetachOption = False,
) -> None:
    config = resolve_workflow_config(
        EvaluateWorkflowSelection(
            artifact_id=artifact_id,
            corpus_id=corpus_id,
            evaluator=evaluator,
            evaluation_window=_evaluation_window(
                evaluation_start,
                evaluation_duration_seconds,
            ),
            delay_seconds=delay_seconds,
            batch_size=batch_size,
        )
    )
    _submit_selected_workflow(
        config=config,
        target=target,
        dependency=dependency,
        detach=detach,
    )


def _evaluation_window(
    evaluation_start: WorkflowEvaluationWindowStartOption,
    evaluation_duration_seconds: WorkflowEvaluationWindowDurationOption,
) -> TimestampWindowSpec | None:
    if evaluation_start is None and evaluation_duration_seconds is None:
        return None
    if evaluation_start is None or evaluation_duration_seconds is None:
        raise ValueError(
            "--evaluation-start and --evaluation-duration-seconds must be provided together"
        )
    return TimestampWindowSpec.model_validate(
        {
            "start": evaluation_start,
            "duration_seconds": evaluation_duration_seconds,
        }
    )


def _report_workflow_submission_event(
    reporter: Reporter,
    event: WorkflowSubmissionEvent,
) -> None:
    provenance = event.provenance
    if event.kind == "submitted":
        reporter.header(
            "submit",
            [
                ("workflow", provenance.task.value),
                ("job_id", provenance.job_id),
                ("log", provenance.log_path),
            ],
        )
        return
    if event.kind == "detached":
        reporter.header(
            "submit detached",
            [
                ("job_id", provenance.job_id),
                ("state", event.state or "running"),
            ],
        )
        return
    if event.state is None:
        return
    reporter.header(
        "submit finished",
        [
            ("job_id", provenance.job_id),
            ("state", event.state),
        ],
    )

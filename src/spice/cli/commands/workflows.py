# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

import typer

from ...config import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
)
from ...config.resolution import resolve_workflow_config
from ...config.resolved_workflows import ResolvedWorkflowConfig
from ...core.errors import SpiceOperatorError
from ...execution.session import ExecutionSession, open_execution_session
from ..options import (
    DEFAULT_REMOTE_TARGET,
    RemoteTargetOption,
    WorkflowArtifactConsumerOption,
    WorkflowBatchSizeOption,
    WorkflowChainOption,
    WorkflowDatasetConsumerOption,
    WorkflowDelaySecondsOption,
    WorkflowDependencyOption,
    WorkflowDetachOption,
    WorkflowDryRunOption,
    WorkflowEvaluationDatasetOption,
    WorkflowEvaluationOverrideOption,
    WorkflowEvaluationSpecOption,
    WorkflowFeaturesOption,
    WorkflowModelOption,
    WorkflowObjectiveOption,
    WorkflowProblemOption,
    WorkflowProviderOption,
    WorkflowSplitOption,
    WorkflowStorageRootWriteOption,
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
    session: ExecutionSession,
    dependency: str | None,
    detach: bool,
) -> None:
    task = config.workflow
    submission = session.submit_workflow(
        task,
        config=config,
        dependency=dependency,
    )
    typer.echo(
        " ".join(
            [
                "submit",
                f"workflow={task.value}",
                f"job_id={submission.job_id}",
                f"log={submission.log_path}",
            ]
        )
    )
    if detach or not session.follow_by_default:
        return
    try:
        state = session.follow_job(submission)
    except KeyboardInterrupt:
        typer.echo(f"submit detached job_id={submission.job_id} state=running")
        return
    if state is None:
        return
    typer.echo(f"submit finished job_id={submission.job_id} state={state}")
    if state != "COMPLETED":
        raise SpiceOperatorError(f"Job {submission.job_id} ended with state {state}")


def _submit_model_workflow(
    *,
    selection: TrainWorkflowSelection | TuneWorkflowSelection,
    target: str,
    dependency: str | None,
    detach: bool,
) -> None:
    config = resolve_workflow_config(selection)
    session = open_execution_session(target)
    _submit_selected_workflow(
        config=config,
        session=session,
        dependency=dependency,
        detach=detach,
    )


def acquire_command(
    surface: WorkflowSurfaceOption = None,
    chain: WorkflowChainOption = None,
    problem: WorkflowProblemOption = None,
    features: WorkflowFeaturesOption = None,
    provider: WorkflowProviderOption = None,
    storage_root: WorkflowStorageRootWriteOption = None,
    dry_run: WorkflowDryRunOption = None,
) -> None:
    from ...workflows import acquire

    acquire.run(
        resolve_workflow_config(
            AcquireWorkflowSelection(
                surface=surface,
                chain=chain,
                problem=problem,
                features=features,
                provider=provider,
                storage_root=storage_root,
                dry_run=dry_run,
            )
        )
    )


def train_command(
    surface: WorkflowSurfaceOption = None,
    chain: WorkflowChainOption = None,
    problem: WorkflowProblemOption = None,
    features: WorkflowFeaturesOption = None,
    objective: WorkflowObjectiveOption = None,
    evaluation: WorkflowEvaluationOverrideOption = None,
    model: WorkflowModelOption = None,
    tuning_space: WorkflowTuningSpaceOption = None,
    training: WorkflowTrainingOption = None,
    split: WorkflowSplitOption = None,
    tuning: WorkflowTuningOption = None,
    study: WorkflowStudyOption = None,
    variant: WorkflowVariantOption = None,
    dataset_id: WorkflowDatasetConsumerOption = None,
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
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            dataset_id=dataset_id,
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
    objective: WorkflowObjectiveOption = None,
    evaluation: WorkflowEvaluationOverrideOption = None,
    model: WorkflowModelOption = None,
    tuning_space: WorkflowTuningSpaceOption = None,
    training: WorkflowTrainingOption = None,
    split: WorkflowSplitOption = None,
    tuning: WorkflowTuningOption = None,
    study: WorkflowStudyOption = None,
    dataset_id: WorkflowDatasetConsumerOption = None,
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
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            dataset_id=dataset_id,
            trial_count=trial_count,
        ),
        target=target,
        dependency=dependency,
        detach=detach,
    )


def evaluate_command(
    artifact_id: WorkflowArtifactConsumerOption = None,
    dataset_id: WorkflowEvaluationDatasetOption = None,
    evaluation: WorkflowEvaluationSpecOption = None,
    delay_seconds: WorkflowDelaySecondsOption = None,
    batch_size: WorkflowBatchSizeOption = None,
    dependency: WorkflowDependencyOption = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: WorkflowDetachOption = False,
) -> None:
    config = resolve_workflow_config(
        EvaluateWorkflowSelection(
            artifact_id=artifact_id,
            dataset_id=dataset_id,
            evaluation=evaluation,
            delay_seconds=delay_seconds,
            batch_size=batch_size,
        )
    )
    session = open_execution_session(target)
    _submit_selected_workflow(
        config=config,
        session=session,
        dependency=dependency,
        detach=detach,
    )

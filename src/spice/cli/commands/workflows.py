# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ...config.models import AcquireConfig, WorkflowTask
from ...config.resolution import WorkflowConfig, resolve_workflow_config
from ...config.selections import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    WorkflowSelection,
    workflow_selection_payload,
)
from ...core.errors import SpiceOperatorError
from ...execution.session import ExecutionSession, open_execution_session
from ..options import DEFAULT_REMOTE_TARGET, RemoteTargetOption

ModelWorkflowSelectionType = type[TrainWorkflowSelection] | type[TuneWorkflowSelection]
ModelWorkflowSelection = TrainWorkflowSelection | TuneWorkflowSelection


def _selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def _execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _output_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Outputs")


def _submit_selected_workflow(
    *,
    task: WorkflowTask,
    config: WorkflowConfig,
    session: ExecutionSession,
    dependency: str | None,
    detach: bool,
) -> None:
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
    if detach or not submission.target.spec.follow_by_default:
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


def _resolve_selection_for_task(
    task: WorkflowTask,
    selection: WorkflowSelection,
) -> WorkflowConfig:
    return resolve_workflow_config(task, selection)


def _build_acquire_workflow_config(
    *,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    provider: str | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> AcquireConfig:
    config = _resolve_selection_for_task(
        WorkflowTask.ACQUIRE,
        AcquireWorkflowSelection(
            surface=surface,
            chain=chain,
            problem=problem,
            features=features,
            provider=provider,
            storage_root=storage_root,
            dry_run=dry_run,
        ),
    )
    if not isinstance(config, AcquireConfig):
        raise TypeError("acquire selection resolved to non-acquire config")
    return config


def _build_model_workflow_selection(
    workflow: WorkflowTask,
    *,
    selection_type: ModelWorkflowSelectionType,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    dataset_id: str | None = None,
    study_id: str | None = None,
    variant: str | None = None,
    trial_count: int | None = None,
) -> ModelWorkflowSelection:
    return selection_type.model_validate(
        workflow_selection_payload(
            workflow,
            {
                "surface": surface,
                "chain": chain,
                "problem": problem,
                "features": features,
                "objective": objective,
                "evaluation": evaluation,
                "model": model,
                "tuning_space": tuning_space,
                "training": training,
                "split": split,
                "tuning": tuning,
                "study": study,
                "dataset_id": dataset_id,
                "study_id": study_id,
                "variant": variant,
                "trial_count": trial_count,
                "storage_root": None,
            },
        )
    )


def _build_model_workflow_config(
    *,
    task: WorkflowTask,
    selection_type: ModelWorkflowSelectionType,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    dataset_id: str | None = None,
    study_id: str | None = None,
    variant: str | None = None,
    trial_count: int | None = None,
) -> WorkflowConfig:
    return _resolve_selection_for_task(
        task,
        _build_model_workflow_selection(
            task,
            selection_type=selection_type,
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
            trial_count=trial_count,
        ),
    )


def _build_evaluate_workflow_config(
    *,
    artifact_id: str | None,
    dataset_id: str | None,
    evaluation: str | None,
    delay_seconds: int | None,
    batch_size: int | None,
) -> WorkflowConfig:
    return _resolve_selection_for_task(
        WorkflowTask.EVALUATE,
        EvaluateWorkflowSelection(
            storage_root=None,
            artifact_id=artifact_id,
            dataset_id=dataset_id,
            evaluation=evaluation,
            delay_seconds=delay_seconds,
            batch_size=batch_size,
        ),
    )


def _submit_model_workflow(
    *,
    task: WorkflowTask,
    selection_type: type[TrainWorkflowSelection] | type[TuneWorkflowSelection],
    target: str,
    dependency: str | None,
    detach: bool,
    surface: str | None,
    chain: str | None,
    problem: str | None,
    features: str | None,
    objective: str | None,
    evaluation: str | None,
    model: str | None,
    tuning_space: str | None,
    training: str | None,
    split: str | None,
    tuning: str | None,
    study: str | None,
    dataset_id: str | None = None,
    study_id: str | None = None,
    variant: str | None = None,
    trial_count: int | None = None,
) -> None:
    config = _build_model_workflow_config(
        task=task,
        selection_type=selection_type,
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
        trial_count=trial_count,
    )
    session = open_execution_session(target)
    _submit_selected_workflow(
        task=task,
        config=config,
        session=session,
        dependency=dependency,
        detach=detach,
    )


def acquire_command(
    surface: Annotated[
        str | None,
        _selection_option(
            "--surface",
            metavar="SURFACE",
            help="Resolve a named workflow surface.",
        ),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Override the problem spec."),
    ] = None,
    features: Annotated[
        str | None,
        _selection_option(
            "--features",
            metavar="FEATURES",
            help="Override the features spec.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        _selection_option(
            "--provider",
            metavar="PROVIDER",
            help="Override the RPC provider spec.",
        ),
    ] = None,
    storage_root: Annotated[
        Path | None,
        _output_option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
        ),
    ] = None,
    dry_run: Annotated[
        bool | None,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Skip persistence and RPC side effects.",
            rich_help_panel="Execution",
        ),
    ] = None,
) -> None:
    from ...workflows import acquire

    acquire.run(
        _build_acquire_workflow_config(
            surface=surface,
            chain=chain,
            problem=problem,
            features=features,
            provider=provider,
            storage_root=storage_root,
            dry_run=dry_run,
        )
    )


def train_command(
    surface: Annotated[
        str | None,
        _selection_option(
            "--surface",
            metavar="SURFACE",
            help="Resolve a named workflow surface.",
        ),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Override the problem spec."),
    ] = None,
    features: Annotated[
        str | None,
        _selection_option(
            "--features",
            metavar="FEATURES",
            help="Override the features spec.",
        ),
    ] = None,
    objective: Annotated[
        str | None,
        _selection_option(
            "--objective",
            metavar="OBJECTIVE",
            help="Override the objective spec.",
        ),
    ] = None,
    evaluation: Annotated[
        str | None,
        _selection_option(
            "--evaluation",
            metavar="EVALUATION",
            help="Override the evaluation spec.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        _selection_option("--model", metavar="MODEL", help="Override the model spec."),
    ] = None,
    tuning_space: Annotated[
        str | None,
        _selection_option(
            "--tuning-space",
            metavar="TUNING_SPACE",
            help="Override the tuning-space spec.",
        ),
    ] = None,
    training: Annotated[
        str | None,
        _selection_option(
            "--training",
            metavar="TRAINING",
            help="Override the training spec.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        _selection_option("--split", metavar="SPLIT", help="Override the split spec."),
    ] = None,
    tuning: Annotated[
        str | None,
        _selection_option("--tuning", metavar="TUNING", help="Override the tuning spec."),
    ] = None,
    study: Annotated[
        str | None,
        _selection_option("--study", metavar="STUDY", help="Override the study name."),
    ] = None,
    variant: Annotated[
        str | None,
        _selection_option("--variant", metavar="VARIANT", help="Override the artifact variant."),
    ] = None,
    dataset_id: Annotated[
        str | None,
        _selection_option("--dataset-id", metavar="DATASET_ID", help="Consume this corpus root."),
    ] = None,
    study_id: Annotated[
        str | None,
        _selection_option("--study-id", metavar="STUDY_ID", help="Consume this study root."),
    ] = None,
    dependency: Annotated[
        str | None,
        _execution_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    _submit_model_workflow(
        task=WorkflowTask.TRAIN,
        selection_type=TrainWorkflowSelection,
        target=target,
        dependency=dependency,
        detach=detach,
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
    )


def tune_command(
    surface: Annotated[
        str | None,
        _selection_option(
            "--surface",
            metavar="SURFACE",
            help="Resolve a named workflow surface.",
        ),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Override the problem spec."),
    ] = None,
    features: Annotated[
        str | None,
        _selection_option(
            "--features",
            metavar="FEATURES",
            help="Override the features spec.",
        ),
    ] = None,
    objective: Annotated[
        str | None,
        _selection_option(
            "--objective",
            metavar="OBJECTIVE",
            help="Override the objective spec.",
        ),
    ] = None,
    evaluation: Annotated[
        str | None,
        _selection_option(
            "--evaluation",
            metavar="EVALUATION",
            help="Override the evaluation spec.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        _selection_option("--model", metavar="MODEL", help="Override the model spec."),
    ] = None,
    tuning_space: Annotated[
        str | None,
        _selection_option(
            "--tuning-space",
            metavar="TUNING_SPACE",
            help="Override the tuning-space spec.",
        ),
    ] = None,
    training: Annotated[
        str | None,
        _selection_option(
            "--training",
            metavar="TRAINING",
            help="Override the training spec.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        _selection_option("--split", metavar="SPLIT", help="Override the split spec."),
    ] = None,
    tuning: Annotated[
        str | None,
        _selection_option("--tuning", metavar="TUNING", help="Override the tuning spec."),
    ] = None,
    study: Annotated[
        str | None,
        _selection_option("--study", metavar="STUDY", help="Override the study name."),
    ] = None,
    dataset_id: Annotated[
        str | None,
        _selection_option("--dataset-id", metavar="DATASET_ID", help="Consume this corpus root."),
    ] = None,
    trial_count: Annotated[
        int | None,
        _execution_option(
            "--trial-count",
            metavar="COUNT",
            help="Override the requested trial count.",
        ),
    ] = None,
    dependency: Annotated[
        str | None,
        _execution_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    _submit_model_workflow(
        task=WorkflowTask.TUNE,
        selection_type=TuneWorkflowSelection,
        target=target,
        dependency=dependency,
        detach=detach,
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
    )


def evaluate_command(
    artifact_id: Annotated[
        str | None,
        _selection_option(
            "--artifact-id",
            metavar="ARTIFACT_ID",
            help="Consume this artifact root.",
        ),
    ] = None,
    dataset_id: Annotated[
        str | None,
        _selection_option(
            "--dataset-id",
            metavar="DATASET_ID",
            help="Evaluate on this corpus root.",
        ),
    ] = None,
    evaluation: Annotated[
        str | None,
        _selection_option("--evaluation", metavar="EVALUATION", help="Use this evaluator spec."),
    ] = None,
    delay_seconds: Annotated[
        int | None,
        _execution_option(
            "--delay-seconds",
            metavar="SECONDS",
            help="Override the evaluation delay in seconds.",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        _execution_option(
            "--batch-size",
            metavar="COUNT",
            help="Override the evaluation batch size.",
        ),
    ] = None,
    dependency: Annotated[
        str | None,
        _execution_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    config = _build_evaluate_workflow_config(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        evaluation=evaluation,
        delay_seconds=delay_seconds,
        batch_size=batch_size,
    )
    session = open_execution_session(target)
    _submit_selected_workflow(
        task=WorkflowTask.EVALUATE,
        config=config,
        session=session,
        dependency=dependency,
        detach=detach,
    )

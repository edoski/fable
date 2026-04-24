# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from ...config.models import WorkflowTask
from ...config.resolution import (
    AcquireWorkflowRequest,
    EvaluateWorkflowRequest,
    TrainWorkflowRequest,
    TuneWorkflowRequest,
    WorkflowConfig,
    WorkflowConfigRequest,
    resolve_workflow_config,
    workflow_request_payload,
)
from ...core.errors import SpiceOperatorError
from ...execution.slurm_ssh import follow_execution_job, submit_execution_workflow


def _selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def _workflow_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _output_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Outputs")


def _submission_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _submit_selected_workflow(
    *,
    task: WorkflowTask,
    request: WorkflowConfigRequest,
    target_name: str,
    dependency: str | None,
    detach: bool,
) -> None:
    config = _resolve_request_for_task(task, request)
    submission = submit_execution_workflow(
        task,
        config=config,
        target_name=target_name,
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
        state = follow_execution_job(submission)
    except KeyboardInterrupt:
        typer.echo(f"submit detached job_id={submission.job_id} state=running")
        return
    if state is None:
        return
    typer.echo(f"submit finished job_id={submission.job_id} state={state}")
    if state != "COMPLETED":
        raise SpiceOperatorError(f"Job {submission.job_id} ended with state {state}")


def _validate_submission_flags(
    *,
    submit: bool,
    dependency: str | None,
    detach: bool,
    storage_root: Path | None,
) -> None:
    if submit:
        if storage_root is not None:
            raise SpiceOperatorError("--storage-root cannot be combined with --submit")
        return
    if dependency is not None or detach:
        raise SpiceOperatorError("--dependency and --detach require --submit")


def _target_option() -> object:
    return _submission_option(
        "--target",
        metavar="TARGET",
        help="Submit to a named execution target.",
    )


def _run_resolved_workflow(
    *,
    task: WorkflowTask,
    runner: Callable[[Any], None],
    request: WorkflowConfigRequest,
) -> None:
    runner(_resolve_request_for_task(task, request))


def _resolve_request_for_task(
    task: WorkflowTask,
    request: WorkflowConfigRequest,
) -> WorkflowConfig:
    return resolve_workflow_config(task, request)


def _request_payload(
    workflow: WorkflowTask,
    **values: object | None,
) -> dict[str, object]:
    return workflow_request_payload(workflow, values)


def _model_request_payload(
    workflow: WorkflowTask,
    *,
    surface: str | None,
    chain: str | None,
    problem: str | None = None,
    feature_set: str | None = None,
    objective: str | None = None,
    evaluation: str | None = None,
    model: str | None = None,
    tuning_space: str | None = None,
    training: str | None = None,
    split: str | None = None,
    tuning: str | None = None,
    study: str | None = None,
    variant: str | None = None,
    delay_seconds: int | None = None,
    trial_count: int | None = None,
    storage_root: Path | None = None,
) -> dict[str, object]:
    return _request_payload(
        workflow,
        surface=surface,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        objective=objective,
        evaluation=evaluation,
        model=model,
        tuning_space=tuning_space,
        training=training,
        split=split,
        tuning=tuning,
        study=study,
        variant=variant,
        delay_seconds=delay_seconds,
        trial_count=trial_count,
        storage_root=storage_root,
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
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Override the feature-set spec.",
        ),
    ] = None,
    acquisition: Annotated[
        str | None,
        _selection_option(
            "--acquisition",
            metavar="ACQUISITION",
            help="Override the acquisition spec.",
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
        resolve_workflow_config(
            WorkflowTask.ACQUIRE,
            AcquireWorkflowRequest(
                surface=surface,
                chain=chain,
                problem=problem,
                feature_set=feature_set,
                acquisition=acquisition,
                storage_root=storage_root,
                dry_run=dry_run,
            ),
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
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Override the feature-set spec.",
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
    storage_root: Annotated[
        Path | None,
        _output_option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
        ),
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the workflow to the checked-in execution target.",
            rich_help_panel="Execution",
        ),
    ] = False,
    dependency: Annotated[
        str | None,
        _submission_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: Annotated[str, _target_option()] = "disi_l40",
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    from ...workflows import train

    _validate_submission_flags(
        submit=submit,
        dependency=dependency,
        detach=detach,
        storage_root=storage_root,
    )
    submit_request = TrainWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.TRAIN,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            variant=variant,
        )
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.TRAIN,
            request=submit_request,
            target_name=target,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = TrainWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.TRAIN,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            variant=variant,
            storage_root=storage_root,
        )
    )
    _run_resolved_workflow(
        task=WorkflowTask.TRAIN,
        runner=train.run,
        request=local_request,
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
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Override the feature-set spec.",
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
    trial_count: Annotated[
        int | None,
        _workflow_option(
            "--trial-count",
            metavar="COUNT",
            help="Override the requested trial count.",
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
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the workflow to the checked-in execution target.",
            rich_help_panel="Execution",
        ),
    ] = False,
    dependency: Annotated[
        str | None,
        _submission_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: Annotated[str, _target_option()] = "disi_l40",
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    from ...workflows import tune

    _validate_submission_flags(
        submit=submit,
        dependency=dependency,
        detach=detach,
        storage_root=storage_root,
    )
    submit_request = TuneWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.TUNE,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            trial_count=trial_count,
        )
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.TUNE,
            request=submit_request,
            target_name=target,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = TuneWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.TUNE,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            trial_count=trial_count,
            storage_root=storage_root,
        )
    )
    _run_resolved_workflow(
        task=WorkflowTask.TUNE,
        runner=tune.run,
        request=local_request,
    )


def evaluate_command(
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
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Override the feature-set spec.",
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
    delay_seconds: Annotated[
        int | None,
        _workflow_option(
            "--delay-seconds",
            metavar="SECONDS",
            help="Override the evaluation delay in seconds.",
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
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the workflow to the checked-in execution target.",
            rich_help_panel="Execution",
        ),
    ] = False,
    dependency: Annotated[
        str | None,
        _submission_option(
            "--dependency",
            metavar="DEPENDENCY",
            help="Pass one Slurm dependency spec such as afterok:12345.",
        ),
    ] = None,
    target: Annotated[str, _target_option()] = "disi_l40",
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Submit and exit without following the job.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    from ...workflows import evaluate

    _validate_submission_flags(
        submit=submit,
        dependency=dependency,
        detach=detach,
        storage_root=storage_root,
    )
    submit_request = EvaluateWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.EVALUATE,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            variant=variant,
            delay_seconds=delay_seconds,
        )
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.EVALUATE,
            request=submit_request,
            target_name=target,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = EvaluateWorkflowRequest.model_validate(
        _model_request_payload(
            WorkflowTask.EVALUATE,
            surface=surface,
            chain=chain,
            problem=problem,
            feature_set=feature_set,
            objective=objective,
            evaluation=evaluation,
            model=model,
            tuning_space=tuning_space,
            training=training,
            split=split,
            tuning=tuning,
            study=study,
            variant=variant,
            delay_seconds=delay_seconds,
            storage_root=storage_root,
        )
    )
    _run_resolved_workflow(
        task=WorkflowTask.EVALUATE,
        runner=evaluate.run,
        request=local_request,
    )

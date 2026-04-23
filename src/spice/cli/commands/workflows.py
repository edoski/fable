# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ...config.models import WorkflowTask
from ...config.resolution import WorkflowRequest, resolve_workflow_config
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
    request: WorkflowRequest,
    dependency: str | None,
    detach: bool,
) -> None:
    resolve_workflow_config(task, request)
    submission = submit_execution_workflow(
        task,
        request=request,
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


def _run_resolved_workflow(
    *,
    task: WorkflowTask,
    runner,
    request: WorkflowRequest,
) -> None:
    runner(resolve_workflow_config(task, request))


def _build_model_workflow_request(
    *,
    preset: str | None,
    chain: str | None,
    problem: str | None = None,
    feature_set: str | None = None,
    evaluation: str | None = None,
    study: str | None = None,
    variant: str | None = None,
    delay_seconds: int | None = None,
    trial_count: int | None = None,
    storage_root: Path | None = None,
) -> WorkflowRequest:
    return WorkflowRequest(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        evaluation=evaluation,
        study=study,
        variant=variant,
        delay_seconds=delay_seconds,
        trial_count=trial_count,
        storage_root=storage_root,
    )


def acquire_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Resolve a named workflow preset.",
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
            WorkflowRequest(
                preset=preset,
                chain=chain,
                problem=problem,
                feature_set=feature_set,
                storage_root=storage_root,
                dry_run=dry_run,
            ),
        )
    )


def train_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Resolve a named workflow preset.",
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
    submit_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        study=study,
        variant=variant,
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.TRAIN,
            request=submit_request,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        study=study,
        variant=variant,
        storage_root=storage_root,
    )
    _run_resolved_workflow(
        task=WorkflowTask.TRAIN,
        runner=train.run,
        request=local_request,
    )


def tune_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Resolve a named workflow preset.",
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
    submit_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        trial_count=trial_count,
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.TUNE,
            request=submit_request,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        trial_count=trial_count,
        storage_root=storage_root,
    )
    _run_resolved_workflow(
        task=WorkflowTask.TUNE,
        runner=tune.run,
        request=local_request,
    )


def evaluate_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Resolve a named workflow preset.",
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
    evaluation: Annotated[
        str | None,
        _selection_option(
            "--evaluation",
            metavar="EVALUATION",
            help="Override the evaluation spec.",
        ),
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
    submit_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        evaluation=evaluation,
        study=study,
        variant=variant,
        delay_seconds=delay_seconds,
    )
    if submit:
        _submit_selected_workflow(
            task=WorkflowTask.EVALUATE,
            request=submit_request,
            dependency=dependency,
            detach=detach,
        )
        return
    local_request = _build_model_workflow_request(
        preset=preset,
        chain=chain,
        problem=problem,
        feature_set=feature_set,
        evaluation=evaluation,
        study=study,
        variant=variant,
        delay_seconds=delay_seconds,
        storage_root=storage_root,
    )
    _run_resolved_workflow(
        task=WorkflowTask.EVALUATE,
        runner=evaluate.run,
        request=local_request,
    )

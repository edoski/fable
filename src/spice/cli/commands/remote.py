"""Explicit remote CLI routing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ...config import WorkflowTask
from ...config.registry import ConfigGroup, load_named_group
from ...core.errors import ConfigResolutionError, SpiceOperatorError
from ...remote import (
    DEFAULT_REMOTE_EXECUTION_NAME,
    follow_remote_job,
    resolve_remote_target,
    run_remote_cli,
    submit_remote_workflow,
)
from ..options import (
    ChainFilterOption,
    DatasetFilterOption,
    FeatureSetFilterOption,
    ModelFilterOption,
    PredictionFilterOption,
    ProblemFilterOption,
    StudyFilterOption,
    VariantFilterOption,
)

remote_app = typer.Typer(
    help="Run workflows and queries against the remote cluster.",
    no_args_is_help=True,
)
config_app = typer.Typer(
    help="Query saved YAML config specs on the remote cluster.",
    no_args_is_help=True,
)
show_app = typer.Typer(
    help="Query remote datasets, studies, and artifacts.",
    no_args_is_help=True,
)
refresh_app = typer.Typer(
    help="Rebuild derived remote storage indexes.",
    no_args_is_help=True,
)
remote_app.add_typer(config_app, name="config")
remote_app.add_typer(show_app, name="show")
remote_app.add_typer(refresh_app, name="refresh")

_CONFIG_GROUP_HELP = (
    "One of: chain, dataset, execution, feature-set, model, prediction, preset, problem, "
    "provider, tuning-space."
)


def _selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def _execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _append_option(args: list[str], flag: str, value: str | int | Path | None) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def _build_cli_args(*options: tuple[str, str | int | Path | None]) -> list[str]:
    args: list[str] = []
    for flag, value in options:
        _append_option(args, flag, value)
    return args


def _resolve_remote_execution_name(*, preset: str | None, execution: str | None) -> str:
    if execution is not None:
        return execution
    if preset is not None:
        execution_value = load_named_group(preset, "preset").get("execution")
        if execution_value is None:
            return DEFAULT_REMOTE_EXECUTION_NAME
        if isinstance(execution_value, str):
            return execution_value
        raise ConfigResolutionError("preset.execution must be a string")
    return DEFAULT_REMOTE_EXECUTION_NAME


def _submit_remote_selected_workflow(
    *,
    task: WorkflowTask,
    preset: str | None,
    execution: str | None,
    detach: bool,
    cli_options: list[tuple[str, str | int | Path | None]],
) -> None:
    submission = submit_remote_workflow(
        task,
        cli_args=_build_cli_args(*cli_options),
        execution_name=_resolve_remote_execution_name(preset=preset, execution=execution),
    )
    typer.echo(
        " ".join(
            [
                f"submitted remote {task.value}",
                f"job_id={submission.job_id}",
                f"execution={submission.execution_name}",
                f"log={submission.log_path}",
            ]
        )
    )
    if detach or not submission.target.spec.follow_by_default:
        return
    try:
        state = follow_remote_job(submission)
    except KeyboardInterrupt:
        typer.echo(f"detached from remote job {submission.job_id}; job continues on cluster")
        return
    if state is not None:
        typer.echo(f"remote job {submission.job_id} finished: {state}")
        if state != "COMPLETED":
            raise SpiceOperatorError(f"Remote job {submission.job_id} ended with state {state}")


def _run_remote_cli(args: list[str], *, error_message: str) -> None:
    result = run_remote_cli(resolve_remote_target(), args)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise SpiceOperatorError(message or error_message)
    typer.echo(result.stdout, nl=False)


def _run_remote_storage_cli(
    args: list[str],
    *,
    error_message: str,
) -> None:
    target = resolve_remote_target()
    result = run_remote_cli(
        target,
        [
            *args,
            "--storage-root",
            str(target.spec.paths.storage_root),
        ],
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise SpiceOperatorError(message or error_message)
    typer.echo(result.stdout, nl=False)


@remote_app.command(
    "train",
    short_help="Train a model artifact on the remote cluster.",
    help="Submit one training workflow to the configured remote execution target.",
)
def remote_train_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Apply a named preset before selector overrides.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        _selection_option("--dataset", metavar="DATASET", help="Use a named dataset spec."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Use a named problem spec."),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    model: Annotated[
        str | None,
        _selection_option("--model", metavar="MODEL", help="Use a named model config."),
    ] = None,
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature selection.",
        ),
    ] = None,
    prediction: Annotated[
        str | None,
        _selection_option(
            "--prediction",
            metavar="PREDICTION",
            help="Use a named prediction config.",
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
    execution: Annotated[
        str | None,
        _execution_option(
            "--execution",
            metavar="EXECUTION",
            help="Use a named remote execution spec.",
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option("--detach", help="Submit remote job and exit without following."),
    ] = False,
) -> None:
    _submit_remote_selected_workflow(
        task=WorkflowTask.TRAIN,
        preset=preset,
        execution=execution,
        detach=detach,
        cli_options=[
            ("--preset", preset),
            ("--dataset", dataset),
            ("--problem", problem),
            ("--chain", chain),
            ("--model", model),
            ("--feature-set", feature_set),
            ("--prediction", prediction),
            ("--study", study),
            ("--variant", variant),
        ],
    )


@remote_app.command(
    "tune",
    short_help="Tune a model artifact on the remote cluster.",
    help="Submit one tuning workflow to the configured remote execution target.",
)
def remote_tune_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Apply a named preset before selector overrides.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        _selection_option("--dataset", metavar="DATASET", help="Use a named dataset spec."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Use a named problem spec."),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    model: Annotated[
        str | None,
        _selection_option("--model", metavar="MODEL", help="Use a named model config."),
    ] = None,
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature selection.",
        ),
    ] = None,
    prediction: Annotated[
        str | None,
        _selection_option(
            "--prediction",
            metavar="PREDICTION",
            help="Use a named prediction config.",
        ),
    ] = None,
    study: Annotated[
        str | None,
        _selection_option("--study", metavar="STUDY", help="Override the study name."),
    ] = None,
    trial_count: Annotated[
        int | None,
        _execution_option(
            "--trial-count",
            metavar="COUNT",
            help="Override the requested trial count.",
        ),
    ] = None,
    execution: Annotated[
        str | None,
        _execution_option(
            "--execution",
            metavar="EXECUTION",
            help="Use a named remote execution spec.",
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option("--detach", help="Submit remote job and exit without following."),
    ] = False,
) -> None:
    _submit_remote_selected_workflow(
        task=WorkflowTask.TUNE,
        preset=preset,
        execution=execution,
        detach=detach,
        cli_options=[
            ("--preset", preset),
            ("--dataset", dataset),
            ("--problem", problem),
            ("--chain", chain),
            ("--model", model),
            ("--feature-set", feature_set),
            ("--prediction", prediction),
            ("--study", study),
            ("--trial-count", trial_count),
        ],
    )


@remote_app.command(
    "evaluate",
    short_help="Evaluate a model artifact on the remote cluster.",
    help="Submit one evaluation workflow to the configured remote execution target.",
)
def remote_evaluate_command(
    preset: Annotated[
        str | None,
        _selection_option(
            "--preset",
            metavar="PRESET",
            help="Apply a named preset before selector overrides.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        _selection_option("--dataset", metavar="DATASET", help="Use a named dataset spec."),
    ] = None,
    problem: Annotated[
        str | None,
        _selection_option("--problem", metavar="PROBLEM", help="Use a named problem spec."),
    ] = None,
    chain: Annotated[
        str | None,
        _selection_option("--chain", metavar="CHAIN", help="Override the target chain."),
    ] = None,
    model: Annotated[
        str | None,
        _selection_option("--model", metavar="MODEL", help="Use a named model config."),
    ] = None,
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature selection.",
        ),
    ] = None,
    prediction: Annotated[
        str | None,
        _selection_option(
            "--prediction",
            metavar="PREDICTION",
            help="Use a named prediction config.",
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
        _execution_option(
            "--delay-seconds",
            metavar="SECONDS",
            help="Override the evaluation delay in seconds.",
        ),
    ] = None,
    execution: Annotated[
        str | None,
        _execution_option(
            "--execution",
            metavar="EXECUTION",
            help="Use a named remote execution spec.",
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option("--detach", help="Submit remote job and exit without following."),
    ] = False,
) -> None:
    _submit_remote_selected_workflow(
        task=WorkflowTask.EVALUATE,
        preset=preset,
        execution=execution,
        detach=detach,
        cli_options=[
            ("--preset", preset),
            ("--dataset", dataset),
            ("--problem", problem),
            ("--chain", chain),
            ("--model", model),
            ("--feature-set", feature_set),
            ("--prediction", prediction),
            ("--study", study),
            ("--variant", variant),
            ("--delay-seconds", delay_seconds),
        ],
    )


@config_app.command("list", short_help="List saved remote config specs.")
def remote_config_list_command(
    group: Annotated[
        ConfigGroup,
        typer.Argument(metavar="GROUP", help=_CONFIG_GROUP_HELP),
    ],
) -> None:
    _run_remote_cli(["config", "list", group.value], error_message="remote config list failed")


@config_app.command("show", short_help="Show one saved remote config spec.")
def remote_config_show_command(
    group: Annotated[
        ConfigGroup,
        typer.Argument(metavar="GROUP", help=_CONFIG_GROUP_HELP),
    ],
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Saved spec name."),
    ],
) -> None:
    _run_remote_cli(
        ["config", "show", group.value, name],
        error_message="remote config show failed",
    )


def _remote_show_args(
    kind: str,
    *,
    chain: str | None = None,
    dataset: str | None = None,
    feature_set: str | None = None,
    prediction: str | None = None,
    model: str | None = None,
    problem: str | None = None,
    variant: str | None = None,
    study: str | None = None,
    detail: str | None = None,
) -> list[str]:
    return _build_cli_args(
        ("show", kind),
        ("--chain", chain),
        ("--dataset", dataset),
        ("--feature-set", feature_set),
        ("--prediction", prediction),
        ("--model", model),
        ("--problem", problem),
        ("--variant", variant),
        ("--study", study),
        ("--detail", detail),
    )


@show_app.command("dataset", short_help="Show remote datasets.")
def remote_show_dataset_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    detail: Annotated[
        str | None,
        typer.Option("--detail", metavar="DETAIL", help="Show one detail table: runs."),
    ] = None,
) -> None:
    _run_remote_storage_cli(
        _remote_show_args("dataset", chain=chain, dataset=dataset, detail=detail),
        error_message="remote show failed",
    )


@show_app.command("study", short_help="Show remote studies.")
def remote_show_study_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    feature_set: FeatureSetFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    study: StudyFilterOption = None,
    detail: Annotated[
        str | None,
        typer.Option(
            "--detail",
            metavar="DETAIL",
            help="Show one detail table: trials or config.",
        ),
    ] = None,
) -> None:
    _run_remote_storage_cli(
        _remote_show_args(
            "study",
            chain=chain,
            dataset=dataset,
            feature_set=feature_set,
            prediction=prediction,
            model=model,
            problem=problem,
            study=study,
            detail=detail,
        ),
        error_message="remote show failed",
    )


@show_app.command("artifact", short_help="Show remote artifacts.")
def remote_show_artifact_command(
    chain: ChainFilterOption = None,
    dataset: DatasetFilterOption = None,
    feature_set: FeatureSetFilterOption = None,
    prediction: PredictionFilterOption = None,
    model: ModelFilterOption = None,
    problem: ProblemFilterOption = None,
    variant: VariantFilterOption = None,
    study: StudyFilterOption = None,
    detail: Annotated[
        str | None,
        typer.Option(
            "--detail",
            metavar="DETAIL",
            help="Show one detail table: epochs or runs.",
        ),
    ] = None,
) -> None:
    _run_remote_storage_cli(
        _remote_show_args(
            "artifact",
            chain=chain,
            dataset=dataset,
            feature_set=feature_set,
            prediction=prediction,
            model=model,
            problem=problem,
            variant=variant,
            study=study,
            detail=detail,
        ),
        error_message="remote show failed",
    )


@refresh_app.command("catalog", short_help="Rebuild the derived remote storage catalog.")
def remote_refresh_catalog_command() -> None:
    target = resolve_remote_target()
    _run_remote_cli(
        [
            "refresh",
            "catalog",
            "--storage-root",
            str(target.spec.paths.storage_root),
        ],
        error_message="remote catalog refresh failed",
    )

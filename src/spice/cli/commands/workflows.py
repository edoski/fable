# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from ...config import WorkflowSelections, WorkflowTask, resolve_workflow_config


def _selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def _execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _output_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Outputs")


def _run_resolved_workflow(
    *,
    task: WorkflowTask,
    runner: Callable[..., None],
    selections: WorkflowSelections,
) -> None:
    runner(resolve_workflow_config(task, selections))


def acquire_command(
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
    provider: Annotated[
        str | None,
        _selection_option("--provider", metavar="PROVIDER", help="Override the RPC provider."),
    ] = None,
    feature_set: Annotated[
        str | None,
        _selection_option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature selection.",
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
            WorkflowSelections(
                preset=preset,
                dataset=dataset,
                problem=problem,
                chain=chain,
                provider=provider,
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
    storage_root: Annotated[
        Path | None,
        _output_option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
        ),
    ] = None,
) -> None:
    from ...workflows import train

    _run_resolved_workflow(
        task=WorkflowTask.TRAIN,
        runner=train.run,
        selections=WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            storage_root=storage_root,
            variant=variant,
            study=study,
        ),
    )


def tune_command(
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
    storage_root: Annotated[
        Path | None,
        _output_option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
        ),
    ] = None,
) -> None:
    from ...workflows import tune

    _run_resolved_workflow(
        task=WorkflowTask.TUNE,
        runner=tune.run,
        selections=WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            storage_root=storage_root,
            study=study,
            trial_count=trial_count,
        ),
    )


def evaluate_command(
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
    storage_root: Annotated[
        Path | None,
        _output_option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
        ),
    ] = None,
) -> None:
    from ...workflows import evaluate

    _run_resolved_workflow(
        task=WorkflowTask.EVALUATE,
        runner=evaluate.run,
        selections=WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            storage_root=storage_root,
            variant=variant,
            study=study,
            delay_seconds=delay_seconds,
        ),
    )

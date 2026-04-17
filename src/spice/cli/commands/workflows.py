# pyright: strict

"""Workflow command routing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ...config import (
    AcquireConfig,
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowSelections,
    WorkflowTask,
    resolve_workflow_config,
)


def _selection_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Selection")


def _execution_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Execution")


def _output_option(*param_decls: str, metavar: str, help: str) -> object:
    return typer.Option(*param_decls, metavar=metavar, help=help, rich_help_panel="Outputs")


def _resolve_acquire_config(
    *,
    preset: str | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    provider: str | None,
    feature_set: str | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> AcquireConfig:
    return resolve_workflow_config(
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


def _resolve_train_config(
    *,
    preset: str | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    study: str | None,
    variant: str | None,
    storage_root: Path | None,
) -> TrainConfig:
    return resolve_workflow_config(
        WorkflowTask.TRAIN,
        WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            study=study,
            variant=variant,
            storage_root=storage_root,
        ),
    )


def _resolve_tune_config(
    *,
    preset: str | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    study: str | None,
    trial_count: int | None,
    storage_root: Path | None,
) -> TuneConfig:
    return resolve_workflow_config(
        WorkflowTask.TUNE,
        WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            study=study,
            trial_count=trial_count,
            storage_root=storage_root,
        ),
    )


def _resolve_evaluate_config(
    *,
    preset: str | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    study: str | None,
    variant: str | None,
    delay_seconds: int | None,
    storage_root: Path | None,
) -> EvaluateConfig:
    return resolve_workflow_config(
        WorkflowTask.EVALUATE,
        WorkflowSelections(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            study=study,
            variant=variant,
            delay_seconds=delay_seconds,
            storage_root=storage_root,
        ),
    )


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
        _resolve_acquire_config(
            preset=preset,
            dataset=dataset,
            problem=problem,
            chain=chain,
            provider=provider,
            feature_set=feature_set,
            storage_root=storage_root,
            dry_run=dry_run,
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

    train.run(
        _resolve_train_config(
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
        )
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

    tune.run(
        _resolve_tune_config(
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
        )
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

    evaluate.run(
        _resolve_evaluate_config(
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
        )
    )

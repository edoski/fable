"""Workflow command routing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _run_acquire(
    *,
    preset: str | None,
    config: Path | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    provider: str | None,
    feature_set: str | None,
    acquisition_profile: str | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> None:
    from ...config import load_acquire_config
    from ...workflows import acquire

    acquire.run(
        load_acquire_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            problem=problem,
            chain=chain,
            provider=provider,
            feature_set=feature_set,
            acquisition=acquisition_profile,
            storage_root=storage_root,
            dry_run=dry_run,
        )
    )


def _run_train(
    *,
    preset: str | None,
    config: Path | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    training_profile: str | None,
    split: str | None,
    storage_root: Path | None,
    variant: str | None,
    study: str | None,
) -> None:
    from ...config import load_train_config
    from ...workflows import train

    train.run(
        load_train_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            training=training_profile,
            split=split,
            storage_root=storage_root,
            variant=variant,
            study=study,
        )
    )


def _run_tune(
    *,
    preset: str | None,
    config: Path | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    training_profile: str | None,
    split: str | None,
    tuning_profile: str | None,
    tuning_space: str | None,
    storage_root: Path | None,
    study: str | None,
    trial_count: int | None,
) -> None:
    from ...config import load_tune_config
    from ...workflows import tune

    tune.run(
        load_tune_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            training=training_profile,
            split=split,
            tuning=tuning_profile,
            tuning_space=tuning_space,
            storage_root=storage_root,
            study=study,
            trial_count=trial_count,
        )
    )


def _run_simulate(
    *,
    preset: str | None,
    config: Path | None,
    dataset: str | None,
    problem: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    prediction: str | None,
    training_profile: str | None,
    simulation_profile: str | None,
    execution: str | None,
    storage_root: Path | None,
    variant: str | None,
    study: str | None,
) -> None:
    from ...config import load_simulate_config
    from ...workflows import simulate

    simulate.run(
        load_simulate_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            problem=problem,
            chain=chain,
            model=model,
            feature_set=feature_set,
            prediction=prediction,
            training=training_profile,
            simulation=simulation_profile,
            execution=execution,
            storage_root=storage_root,
            variant=variant,
            study=study,
        )
    )


def acquire_command(
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            metavar="PRESET",
            help="Apply a named preset before config-file and CLI overrides.",
            rich_help_panel="Selection",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            metavar="PATH",
            help="Overlay a YAML config file on top of the selected preset.",
            rich_help_panel="Overrides",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            metavar="DATASET",
            help="Use a named dataset spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    problem: Annotated[
        str | None,
        typer.Option(
            "--problem",
            metavar="PROBLEM",
            help="Use a named problem profile.",
            rich_help_panel="Selection",
        ),
    ] = None,
    chain: Annotated[
        str | None,
        typer.Option(
            "--chain",
            metavar="CHAIN",
            help="Override the target chain.",
            rich_help_panel="Selection",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            metavar="PROVIDER",
            help="Override the RPC provider.",
            rich_help_panel="Selection",
        ),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature selection.",
            rich_help_panel="Selection",
        ),
    ] = None,
    acquisition_profile: Annotated[
        str | None,
        typer.Option(
            "--acquisition",
            metavar="PROFILE",
            help="Use a named acquisition profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option(
            "--storage-root",
            metavar="PATH",
            help="Store outputs under a non-default root.",
            rich_help_panel="Outputs",
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
    _run_acquire(
        preset=preset,
        config=config,
        dataset=dataset,
        problem=problem,
        chain=chain,
        provider=provider,
        feature_set=feature_set,
        acquisition_profile=acquisition_profile,
        storage_root=storage_root,
        dry_run=dry_run,
    )


def train_command(
    preset: Annotated[
        str | None,
        typer.Option("--preset", metavar="PRESET"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", metavar="PATH"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", metavar="DATASET"),
    ] = None,
    problem: Annotated[
        str | None,
        typer.Option("--problem", metavar="PROBLEM"),
    ] = None,
    chain: Annotated[
        str | None,
        typer.Option("--chain", metavar="CHAIN"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", metavar="MODEL"),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option("--feature-set", metavar="FEATURE_SET"),
    ] = None,
    prediction: Annotated[
        str | None,
        typer.Option("--prediction", metavar="PREDICTION"),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option("--training", metavar="PROFILE"),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", metavar="PROFILE"),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option("--storage-root", metavar="PATH"),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option("--variant", metavar="VARIANT"),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option("--study", metavar="STUDY"),
    ] = None,
) -> None:
    _run_train(
        preset=preset,
        config=config,
        dataset=dataset,
        problem=problem,
        chain=chain,
        model=model,
        feature_set=feature_set,
        prediction=prediction,
        training_profile=training_profile,
        split=split,
        storage_root=storage_root,
        variant=variant,
        study=study,
    )


def tune_command(
    preset: Annotated[
        str | None,
        typer.Option("--preset", metavar="PRESET"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", metavar="PATH"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", metavar="DATASET"),
    ] = None,
    problem: Annotated[
        str | None,
        typer.Option("--problem", metavar="PROBLEM"),
    ] = None,
    chain: Annotated[
        str | None,
        typer.Option("--chain", metavar="CHAIN"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", metavar="MODEL"),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option("--feature-set", metavar="FEATURE_SET"),
    ] = None,
    prediction: Annotated[
        str | None,
        typer.Option("--prediction", metavar="PREDICTION"),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option("--training", metavar="PROFILE"),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", metavar="PROFILE"),
    ] = None,
    tuning_profile: Annotated[
        str | None,
        typer.Option("--tuning", metavar="PROFILE"),
    ] = None,
    tuning_space: Annotated[
        str | None,
        typer.Option("--tuning-space", metavar="PROFILE"),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option("--storage-root", metavar="PATH"),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option("--study", metavar="STUDY"),
    ] = None,
    trial_count: Annotated[
        int | None,
        typer.Option("--trial-count", metavar="N"),
    ] = None,
) -> None:
    _run_tune(
        preset=preset,
        config=config,
        dataset=dataset,
        problem=problem,
        chain=chain,
        model=model,
        feature_set=feature_set,
        prediction=prediction,
        training_profile=training_profile,
        split=split,
        tuning_profile=tuning_profile,
        tuning_space=tuning_space,
        storage_root=storage_root,
        study=study,
        trial_count=trial_count,
    )


def simulate_command(
    preset: Annotated[
        str | None,
        typer.Option("--preset", metavar="PRESET"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", metavar="PATH"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", metavar="DATASET"),
    ] = None,
    problem: Annotated[
        str | None,
        typer.Option("--problem", metavar="PROBLEM"),
    ] = None,
    chain: Annotated[
        str | None,
        typer.Option("--chain", metavar="CHAIN"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", metavar="MODEL"),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option("--feature-set", metavar="FEATURE_SET"),
    ] = None,
    prediction: Annotated[
        str | None,
        typer.Option("--prediction", metavar="PREDICTION"),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option("--training", metavar="PROFILE"),
    ] = None,
    simulation_profile: Annotated[
        str | None,
        typer.Option("--simulation", metavar="PROFILE"),
    ] = None,
    execution: Annotated[
        str | None,
        typer.Option("--execution", metavar="EXECUTION"),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option("--storage-root", metavar="PATH"),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option("--variant", metavar="VARIANT"),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option("--study", metavar="STUDY"),
    ] = None,
) -> None:
    _run_simulate(
        preset=preset,
        config=config,
        dataset=dataset,
        problem=problem,
        chain=chain,
        model=model,
        feature_set=feature_set,
        prediction=prediction,
        training_profile=training_profile,
        simulation_profile=simulation_profile,
        execution=execution,
        storage_root=storage_root,
        variant=variant,
        study=study,
    )

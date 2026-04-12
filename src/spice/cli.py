"""Explicit Typer CLI over the split config system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="spice",
    help="SPICE workflow CLI.",
    epilog="Example:\n  spice acquire --preset icdcs_2026",
    no_args_is_help=True,
    add_completion=True,
)


def _run_acquire(
    *,
    preset: str | None,
    config: Path | None,
    dataset: str | None,
    task: str | None,
    chain: str | None,
    provider: str | None,
    feature_set: str | None,
    acquisition_profile: str | None,
    storage_root: Path | None,
    dry_run: bool | None,
) -> None:
    from .config import load_acquire_config
    from .workflows import acquire

    acquire.run(
        load_acquire_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            task=task,
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
    task: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    training_profile: str | None,
    split: str | None,
    storage_root: Path | None,
    variant: str | None,
    study: str | None,
) -> None:
    from .config import load_train_config
    from .workflows import train

    train.run(
        load_train_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            task=task,
            chain=chain,
            model=model,
            feature_set=feature_set,
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
    task: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    training_profile: str | None,
    split: str | None,
    tuning_profile: str | None,
    tuning_space: str | None,
    storage_root: Path | None,
    study: str | None,
    trial_count: int | None,
) -> None:
    from .config import load_tune_config
    from .workflows import tune

    tune.run(
        load_tune_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            task=task,
            chain=chain,
            model=model,
            feature_set=feature_set,
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
    task: str | None,
    chain: str | None,
    model: str | None,
    feature_set: str | None,
    training_profile: str | None,
    simulation_profile: str | None,
    execution: str | None,
    storage_root: Path | None,
    variant: str | None,
    study: str | None,
) -> None:
    from .config import load_simulate_config
    from .workflows import simulate

    simulate.run(
        load_simulate_config(
            preset=preset,
            config_path=config,
            dataset=dataset,
            task=task,
            chain=chain,
            model=model,
            feature_set=feature_set,
            training=training_profile,
            simulation=simulation_profile,
            execution=execution,
            storage_root=storage_root,
            variant=variant,
            study=study,
        )
    )


def _run_show(
    *,
    root: Path,
    detail: str | None,
    as_json: bool,
) -> None:
    from .core.console import create_console_runtime
    from .state import STATE_DB_FILENAME
    from .state.show import describe_root, sectioned_summary

    target_root = root
    if target_root.is_file() and target_root.name == STATE_DB_FILENAME:
        target_root = target_root.parent.parent
    try:
        payload = describe_root(target_root, detail=detail)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    title, sections = sectioned_summary(payload)
    runtime = create_console_runtime()
    try:
        with runtime.activate():
            runtime.log_sectioned_summary(title, sections)
    finally:
        runtime.close()


@app.command(
    "show",
    short_help="Inspect a dataset, artifact, or study root.",
    help="Inspect one generated state root and print a concise summary.",
    epilog=(
        "Example:\n"
        "  spice show outputs/datasets/avalanche/icdcs_2026\n"
        "  spice show outputs/models/.../tuned/default --detail trials"
    ),
)
def show_command(
    root: Annotated[
        Path,
        typer.Argument(
            metavar="ROOT",
            help="Dataset, artifact, or study root to inspect.",
        ),
    ],
    detail: Annotated[
        str | None,
        typer.Option(
            "--detail",
            metavar="DETAIL",
            help="Show one detail table: trials, epochs, or runs.",
            rich_help_panel="Execution",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the inspected state as JSON to stdout.",
            rich_help_panel="Execution",
        ),
    ] = False,
) -> None:
    _run_show(root=root, detail=detail, as_json=as_json)


@app.command(
    "acquire",
    short_help="Acquire canonical datasets.",
    help="Acquire canonical history and evaluation block datasets.",
    epilog=(
        "Example:\n"
        "  spice acquire --preset icdcs_2026 --chain avalanche --provider publicnode"
    ),
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
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            metavar="TASK",
            help="Use a named task profile.",
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
            help="Use a named feature set profile for acquisition planning.",
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
            help="Write datasets under a non-default output root.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    dry_run: Annotated[
        bool | None,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Plan the acquisition and print windows without writing outputs.",
            rich_help_panel="Execution",
        ),
    ] = None,
) -> None:
    _run_acquire(
        preset=preset,
        config=config,
        dataset=dataset,
        task=task,
        chain=chain,
        provider=provider,
        feature_set=feature_set,
        acquisition_profile=acquisition_profile,
        storage_root=storage_root,
        dry_run=dry_run,
    )


@app.command(
    "train",
    short_help="Train a model artifact.",
    help="Train a model artifact from a canonical history dataset.",
    epilog="Example:\n  spice train --preset icdcs_2026 --variant tuned",
)
def train_command(
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
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            metavar="TASK",
            help="Use a named task profile.",
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
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            metavar="MODEL",
            help="Use a named model spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature-set spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option(
            "--training",
            metavar="PROFILE",
            help="Use a named training profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            metavar="SPLIT",
            help="Use a named train/validation/test split profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option(
            "--storage-root",
            metavar="PATH",
            help="Write artifacts under a non-default output root.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            "--variant",
            metavar="VARIANT",
            help="Select the artifact variant, such as baseline or tuned.",
            rich_help_panel="Execution",
        ),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option(
            "--study",
            metavar="STUDY",
            help="Override the study id used for tuned artifacts.",
            rich_help_panel="Execution",
        ),
    ] = None,
) -> None:
    _run_train(
        preset=preset,
        config=config,
        dataset=dataset,
        task=task,
        chain=chain,
        model=model,
        feature_set=feature_set,
        training_profile=training_profile,
        split=split,
        storage_root=storage_root,
        variant=variant,
        study=study,
    )


@app.command(
    "tune",
    short_help="Tune model hyperparameters.",
    help="Run Optuna tuning for a model and write the best parameter set.",
    epilog="Example:\n  spice tune --preset icdcs_2026 --trial-count 20",
)
def tune_command(
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
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            metavar="TASK",
            help="Use a named task profile.",
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
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            metavar="MODEL",
            help="Use a named model spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature-set spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option(
            "--training",
            metavar="PROFILE",
            help="Use a named training profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            metavar="SPLIT",
            help="Use a named train/validation/test split profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    tuning_profile: Annotated[
        str | None,
        typer.Option(
            "--tuning",
            metavar="PROFILE",
            help="Use a named tuning profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    tuning_space: Annotated[
        str | None,
        typer.Option(
            "--tuning-space",
            metavar="SPACE",
            help="Use a named tuning search space.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option(
            "--storage-root",
            metavar="PATH",
            help="Write tuning outputs under a non-default output root.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option(
            "--study",
            metavar="STUDY",
            help="Override the study id.",
            rich_help_panel="Execution",
        ),
    ] = None,
    trial_count: Annotated[
        int | None,
        typer.Option(
            "--trial-count",
            metavar="COUNT",
            min=1,
            help="Override the number of Optuna trials.",
            rich_help_panel="Execution",
        ),
    ] = None,
) -> None:
    _run_tune(
        preset=preset,
        config=config,
        dataset=dataset,
        task=task,
        chain=chain,
        model=model,
        feature_set=feature_set,
        training_profile=training_profile,
        split=split,
        tuning_profile=tuning_profile,
        tuning_space=tuning_space,
        storage_root=storage_root,
        study=study,
        trial_count=trial_count,
    )


@app.command(
    "simulate",
    short_help="Run evaluation-day simulation.",
    help="Run evaluation-day simulation from a trained artifact.",
    epilog="Example:\n  spice simulate --preset icdcs_2026 --variant tuned",
)
def simulate_command(
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
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            metavar="TASK",
            help="Use a named task profile.",
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
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            metavar="MODEL",
            help="Use a named model spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    feature_set: Annotated[
        str | None,
        typer.Option(
            "--feature-set",
            metavar="FEATURE_SET",
            help="Use a named feature-set spec.",
            rich_help_panel="Selection",
        ),
    ] = None,
    training_profile: Annotated[
        str | None,
        typer.Option(
            "--training",
            metavar="PROFILE",
            help="Use a named training profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    simulation_profile: Annotated[
        str | None,
        typer.Option(
            "--simulation",
            metavar="PROFILE",
            help="Use a named simulation profile.",
            rich_help_panel="Profiles",
        ),
    ] = None,
    execution: Annotated[
        str | None,
        typer.Option(
            "--execution",
            metavar="EXECUTION",
            help="Use a named execution request profile.",
            rich_help_panel="Selection",
        ),
    ] = None,
    storage_root: Annotated[
        Path | None,
        typer.Option(
            "--storage-root",
            metavar="PATH",
            help="Write reports under a non-default output root.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            "--variant",
            metavar="VARIANT",
            help="Select the artifact variant, such as baseline or tuned.",
            rich_help_panel="Execution",
        ),
    ] = None,
    study: Annotated[
        str | None,
        typer.Option(
            "--study",
            metavar="STUDY",
            help="Override the study id used for tuned artifacts.",
            rich_help_panel="Execution",
        ),
    ] = None,
) -> None:
    _run_simulate(
        preset=preset,
        config=config,
        dataset=dataset,
        task=task,
        chain=chain,
        model=model,
        feature_set=feature_set,
        training_profile=training_profile,
        simulation_profile=simulation_profile,
        execution=execution,
        storage_root=storage_root,
        variant=variant,
        study=study,
    )


def main(argv: list[str] | None = None) -> None:
    app(prog_name="spice", args=argv)

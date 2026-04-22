"""Top-level SPICE CLI application."""

from __future__ import annotations

import typer

from .commands.config import app as config_app
from .commands.reconstruct import app as reconstruct_app
from .commands.storage import delete_app, refresh_app, show_app
from .commands.transfer import pull_app, push_app
from .commands.workflows import (
    acquire_command,
    evaluate_command,
    train_command,
    tune_command,
)

app = typer.Typer(
    name="spice",
    help="SPICE workflow CLI.",
    epilog="Example:\n  spice acquire --preset icdcs_2026",
    no_args_is_help=True,
    add_completion=True,
)
app.add_typer(config_app, name="config")
app.add_typer(reconstruct_app, name="reconstruct")
app.add_typer(show_app, name="show")
app.add_typer(delete_app, name="delete")
app.add_typer(push_app, name="push")
app.add_typer(pull_app, name="pull")
app.add_typer(refresh_app, name="refresh")

app.command(
    "acquire",
    short_help="Acquire canonical datasets.",
    help="Acquire canonical history and evaluation block datasets.",
    epilog=(
        "Example:\n"
        "  spice acquire --preset icdcs_2026 --chain avalanche"
    ),
)(acquire_command)
app.command(
    "train",
    short_help="Train a model artifact.",
    help="Train one artifact from a materialized history corpus.",
    epilog=(
        "Example:\n"
        "  spice train --preset icdcs_2026 --study default --variant baseline"
    ),
)(train_command)
app.command(
    "tune",
    short_help="Tune a model artifact.",
    help="Tune one artifact with Optuna.",
    epilog="Example:\n  spice tune --preset icdcs_2026 --trial-count 20",
)(tune_command)
app.command(
    "evaluate",
    short_help="Evaluate a model artifact.",
    help="Evaluate one trained artifact on historical evaluation data.",
    epilog=(
        "Example:\n"
        "  spice evaluate --preset icdcs_2026 --study default --variant baseline"
    ),
)(evaluate_command)


def main(argv: list[str] | None = None) -> None:
    del argv
    app()

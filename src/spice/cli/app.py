"""Top-level SPICE CLI application."""

from __future__ import annotations

import typer

from .commands.benchmark import app as benchmark_app
from .commands.config import app as config_app
from .commands.storage import delete_app, refresh_app, show_app
from .commands.transfer import app as transfer_app
from .commands.workflows import (
    acquire_command,
    evaluate_command,
    train_command,
    tune_command,
)

app = typer.Typer(
    name="spice",
    help="SPICE workflow CLI.",
    epilog="Example:\n  spice acquire --surface current_row_fee_dynamics",
    no_args_is_help=True,
    add_completion=True,
)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(config_app, name="config")
app.add_typer(show_app, name="show")
app.add_typer(delete_app, name="delete")
app.add_typer(transfer_app, name="transfer")
app.add_typer(refresh_app, name="refresh")

app.command(
    "acquire",
    short_help="Acquire canonical datasets.",
    help="Acquire canonical history and evaluation block datasets.",
    epilog=(
        "Example:\n"
        "  spice acquire --surface current_row_fee_dynamics --chain avalanche"
    ),
)(acquire_command)
app.command(
    "train",
    short_help="Train a model artifact.",
    help="Train one artifact from a materialized history corpus.",
    epilog=(
        "Example:\n"
        "  spice train --surface current_row_fee_dynamics "
        "--dataset-id cor_9a73b1e88edb488afb1e"
    ),
)(train_command)
app.command(
    "tune",
    short_help="Tune a model artifact.",
    help="Tune one artifact with Optuna.",
    epilog=(
        "Example:\n"
        "  spice tune --surface current_row_fee_dynamics "
        "--dataset-id cor_9a73b1e88edb488afb1e --trial-count 20"
    ),
)(tune_command)
app.command(
    "evaluate",
    short_help="Evaluate a model artifact.",
    help="Evaluate one trained artifact on historical evaluation data.",
    epilog=(
        "Example:\n"
        "  spice evaluate --artifact-id art_... "
        "--dataset-id cor_9a73b1e88edb488afb1e --evaluation poisson_replay_2h"
    ),
)(evaluate_command)


def main() -> None:
    app()

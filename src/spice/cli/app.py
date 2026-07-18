"""Top-level SPICE CLI application."""

from __future__ import annotations

from .commands.benchmark import app as benchmark_app
from .commands.config import app as config_app
from .commands.corpus import app as corpus_app
from .commands.storage import delete_app, refresh_app, show_app
from .commands.transfer import app as transfer_app
from .commands.workflows import (
    evaluate_command,
    train_command,
    tune_command,
)
from .errors import OperatorTyper

app = OperatorTyper(
    name="spice",
    help="SPICE workflow CLI.",
    epilog="Example:\n  spice corpus acquire REQUEST.json --rpc-url URL --poa",
    no_args_is_help=True,
    add_completion=True,
)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(config_app, name="config")
app.add_typer(corpus_app, name="corpus")
app.add_typer(show_app, name="show")
app.add_typer(delete_app, name="delete")
app.add_typer(transfer_app, name="transfer")
app.add_typer(refresh_app, name="refresh")

app.command(
    "train",
    short_help="Train a model artifact.",
    help="Train one artifact from a materialized history corpus.",
    epilog=(
        "Example:\n"
        "  spice train --surface current_row_fee_dynamics "
        "--corpus-id cor_9a73b1e88edb488afb1e"
    ),
)(train_command)
app.command(
    "tune",
    short_help="Tune a model artifact.",
    help="Tune one artifact with Optuna.",
    epilog=(
        "Example:\n"
        "  spice tune --surface current_row_fee_dynamics "
        "--corpus-id cor_9a73b1e88edb488afb1e --trial-count 20"
    ),
)(tune_command)
app.command(
    "evaluate",
    short_help="Evaluate a model artifact.",
    help="Evaluate one trained artifact on historical evaluation data.",
    epilog=(
        "Example:\n"
        "  spice evaluate --artifact-id art_... "
        "--corpus-id cor_9a73b1e88edb488afb1e --evaluator poisson_replay"
    ),
)(evaluate_command)


def main() -> None:
    app()

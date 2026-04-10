"""Command-line interface for the SPICE temporal baseline project."""

from __future__ import annotations

from pathlib import Path

import typer

from spice_temporal.api import (
    _enrich_blocks,
    _plan_block_pulls,
    _pull_blocks,
    _validate_block_pull,
    run_simulation_workflow,
    run_training_workflow,
)
from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from spice_temporal.config import BlockSegment, ChainName, ModelFamily
from spice_temporal.env import redact_sensitive_text
from spice_temporal.raw_validation import format_raw_pull_validation_report

app = typer.Typer(no_args_is_help=True, add_completion=False)
blocks_app = typer.Typer(no_args_is_help=True)
app.add_typer(blocks_app, name="blocks")


@blocks_app.command("plan")
def plan_blocks(config_path: Path) -> None:
    """Render the cryo commands required for the baseline pull."""
    for plan in _plan_block_pulls(config_path):
        typer.echo(f"[{plan.chain}]")
        typer.echo(f"history={plan.history_range.start}:{plan.history_range.end}")
        typer.echo(f"evaluation={plan.evaluation_range.start}:{plan.evaluation_range.end}")
        typer.echo(plan.command)
        typer.echo("")


@blocks_app.command("enrich")
def enrich_blocks(
    config_path: Path,
    chain_name: ChainName,
    input_path: Path,
    output_path: Path,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    """Add gas_limit to cryo block files using direct JSON-RPC lookups."""
    written = _enrich_blocks(
        config_path,
        chain_name,
        input_path,
        output_path,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )
    typer.echo(f"enriched_files={len(written)}")
    if written:
        typer.echo(f"first_output={written[0]}")


@blocks_app.command("pull")
def pull_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> None:
    """Run cryo for one chain and one dataset segment."""
    try:
        result = _pull_blocks(
            config_path,
            chain_name,
            segment,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if result.process.stdout:
        typer.echo(redact_sensitive_text(result.process.stdout.rstrip()))
    if result.process.stderr:
        typer.echo(redact_sensitive_text(result.process.stderr.rstrip()))
    if result.validation is not None:
        for line in format_raw_pull_validation_report(result.validation):
            typer.echo(line)
        if result.validation.status == "error":
            raise typer.Exit(code=1)


@blocks_app.command("validate")
def validate_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
) -> None:
    """Validate one completed raw block pull without mutating its files."""
    report = _validate_block_pull(config_path, chain_name, segment)
    for line in format_raw_pull_validation_report(report):
        typer.echo(line)
    if report.status == "error":
        raise typer.Exit(code=1)


@app.command("train")
def train(
    config_path: Path,
    history_block_path: Path,
    artifact_dir: Path,
    chain_name: ChainName,
    family: ModelFamily,
    max_delay_seconds: int,
    device: str = "auto",
) -> None:
    """Train one temporal model and write a canonical artifact directory."""
    report = run_training_workflow(
        config_path,
        history_block_path,
        artifact_dir,
        chain_name,
        family,
        max_delay_seconds,
        device=device,
    )
    typer.echo(f"n_blocks_available={report.n_blocks_available}")
    typer.echo(f"n_blocks_used={report.n_blocks_used}")
    typer.echo(f"n_examples_total={report.n_examples_total}")
    typer.echo(f"lookback_steps={report.lookback_steps}")
    typer.echo(f"max_extra_wait_steps={report.max_extra_wait_steps}")
    typer.echo(f"action_count={report.action_count}")
    typer.echo(f"n_features={report.n_features}")
    typer.echo(f"train_examples={report.split_sizes.train_examples}")
    typer.echo(f"validation_examples={report.split_sizes.validation_examples}")
    typer.echo(f"test_examples={report.split_sizes.test_examples}")
    typer.echo(f"best_epoch={report.best_epoch}")
    typer.echo(f"test_loss={report.test_metrics.total_loss:.6f}")
    typer.echo(f"test_accuracy={report.test_metrics.accuracy:.4f}")
    typer.echo(
        f"test_profit_over_baseline={report.test_metrics.mean_profit_over_baseline:.6f}"
    )
    typer.echo(f"artifact_dir={artifact_dir}")
    typer.echo(f"train_report_path={artifact_dir / TRAIN_REPORT_FILENAME}")


@app.command("simulate")
def simulate(
    config_path: Path,
    artifact_dir: Path,
    history_block_path: Path,
    evaluation_block_path: Path,
    device: str = "auto",
) -> None:
    """Run paper-style evaluation-day simulation for a trained artifact."""
    report = run_simulation_workflow(
        config_path,
        artifact_dir,
        history_block_path,
        evaluation_block_path,
        device=device,
    )
    typer.echo(f"n_history_context_blocks={report.n_history_context_blocks}")
    typer.echo(f"n_evaluation_blocks={report.n_evaluation_blocks}")
    typer.echo(f"n_examples_total={report.n_examples_total}")
    typer.echo(f"simulation_profit_over_baseline={report.profit_over_baseline.mean:.6f}")
    typer.echo(f"simulation_cost_over_optimum={report.cost_over_optimum.mean:.6f}")
    typer.echo(
        "simulation_baseline_cost_over_optimum="
        f"{report.baseline_cost_over_optimum.mean:.6f}"
    )
    typer.echo(f"simulation_report_path={artifact_dir / SIMULATION_REPORT_FILENAME}")


if __name__ == "__main__":
    app()

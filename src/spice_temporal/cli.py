"""Command-line interface for the SPICE temporal baseline project."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer

from spice_temporal.config import ExperimentConfig, ModelConfig
from spice_temporal.cryo import build_pull_plan, evaluation_range, history_range_for_chain, run_cryo
from spice_temporal.enrich import enrich_path
from spice_temporal.env import load_project_env, redact_sensitive_text, resolve_rpc_url
from spice_temporal.pipeline import run_single_training
from spice_temporal.rpc import RpcClient

app = typer.Typer(no_args_is_help=True, add_completion=False)
load_project_env()


@app.command("show-config")
def show_config(config_path: Path) -> None:
    """Print a quick summary of the experiment configuration."""
    config = ExperimentConfig.from_yaml(config_path)
    typer.echo(f"Output root: {config.output_root}")
    typer.echo(f"Windows: {config.window_seconds}")
    typer.echo(f"Lookback: {config.lookback_seconds}s")
    typer.echo("Chains:")
    for chain in config.chains:
        typer.echo(
            f"  - {chain.name}: chain_id={chain.chain_id}, "
            f"nominal_block_time={chain.nominal_block_time_seconds}s, "
            f"rpc_env_var={chain.rpc_env_var}"
        )


@app.command("plan-pull")
def plan_pull(config_path: Path) -> None:
    """Render the cryo commands required for the baseline pull."""
    config = ExperimentConfig.from_yaml(config_path)
    for plan in build_pull_plan(config):
        typer.echo(f"[{plan.chain}]")
        typer.echo(f"history={plan.history_range.start}:{plan.history_range.end}")
        typer.echo(f"evaluation={plan.evaluation_range.start}:{plan.evaluation_range.end}")
        typer.echo(plan.command)
        typer.echo("")


@app.command("enrich-blocks")
def enrich_blocks(
    config_path: Path,
    chain_name: str,
    input_path: Path,
    output_path: Path,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    """Add gas_limit to cryo block files using direct JSON-RPC lookups."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = next((item for item in config.chains if item.name == chain_name), None)
    if chain is None:
        raise typer.BadParameter(f"Unknown chain: {chain_name}")
    client = RpcClient(resolve_rpc_url(chain.rpc_env_var))
    written = enrich_path(
        input_path,
        output_path,
        fetch_gas_limits=client.get_block_gas_limits,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
    )
    typer.echo(f"enriched_files={len(written)}")
    if written:
        typer.echo(f"first_output={written[0]}")


@app.command("pull-blocks")
def pull_blocks(
    config_path: Path,
    chain_name: str,
    segment: str,
    dry_run: bool = True,
    overwrite: bool = False,
) -> None:
    """Run cryo for one chain and one dataset segment."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = next((item for item in config.chains if item.name == chain_name), None)
    if chain is None:
        raise typer.BadParameter(f"Unknown chain: {chain_name}")
    if segment not in {"history", "evaluation"}:
        raise typer.BadParameter("segment must be 'history' or 'evaluation'")

    output_dir = config.output_root / "raw" / chain.name / segment
    timestamps = history_range_for_chain(chain) if segment == "history" else evaluation_range()
    result = run_cryo(
        chain,
        output_dir,
        timestamps,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if result.stdout:
        typer.echo(redact_sensitive_text(result.stdout.rstrip()))
    if result.stderr:
        typer.echo(redact_sensitive_text(result.stderr.rstrip()))


@app.command("train-single")
def train_single(
    config_path: Path,
    block_file: Path,
    chain_name: str,
    family: str,
    window_seconds: int,
    device: str = "auto",
) -> None:
    """Run one local training job from a single raw block file."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = next((item for item in config.chains if item.name == chain_name), None)
    if chain is None:
        raise typer.BadParameter(f"Unknown chain: {chain_name}")
    if family not in {"lstm", "transformer", "transformer_lstm"}:
        raise typer.BadParameter(f"Unknown model family: {family}")

    result = run_single_training(
        block_file=block_file,
        chain=chain,
        window_seconds=window_seconds,
        lookback_seconds=config.lookback_seconds,
        model_config=ModelConfig(family=family),
        training_config=replace(config.training, device=device),
        split_config=config.split,
    )
    typer.echo(f"lookback_steps={result.prepared.lookback_steps}")
    typer.echo(f"horizon_blocks={result.prepared.horizon_blocks}")
    typer.echo(f"best_epoch={result.training_result.best_epoch}")
    typer.echo(f"test_loss={result.test_metrics.total_loss:.6f}")
    typer.echo(f"test_accuracy={result.test_metrics.accuracy:.4f}")
    typer.echo(
        f"test_profit_over_baseline={result.test_metrics.mean_profit_over_baseline:.6f}"
    )


if __name__ == "__main__":
    app()

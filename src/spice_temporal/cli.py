"""Command-line interface for the SPICE temporal baseline project."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import typer

from spice_temporal.artifacts import (
    SIMULATION_REPORT_FILENAME,
    TRAIN_REPORT_FILENAME,
    build_training_artifact_manifest,
    load_training_artifact,
    write_training_artifact,
)
from spice_temporal.config import ChainConfig, ExperimentConfig, ModelConfig, ModelFamily
from spice_temporal.cryo import (
    TimestampRange,
    build_pull_plan,
    evaluation_range,
    history_range_for_chain,
    run_cryo,
)
from spice_temporal.datasets import derive_dataset_geometry
from spice_temporal.enrich import enrich_path
from spice_temporal.env import load_project_env, redact_sensitive_text, resolve_rpc_url
from spice_temporal.inference import predict_class_offsets
from spice_temporal.io import load_block_records
from spice_temporal.pipeline import prepare_inference_dataset, run_training
from spice_temporal.raw_validation import (
    format_raw_pull_validation_report,
    validate_raw_pull,
)
from spice_temporal.reporting import (
    build_simulation_report,
    build_training_run_report,
    write_json_report,
)
from spice_temporal.rpc import RpcClient
from spice_temporal.simulation import run_temporal_simulation

app = typer.Typer(no_args_is_help=True, add_completion=False)
load_project_env()

SegmentName = Literal["history", "evaluation"]


def require_chain(config: ExperimentConfig, chain_name: str) -> ChainConfig:
    chain = next((item for item in config.chains if item.name == chain_name), None)
    if chain is None:
        raise typer.BadParameter(f"Unknown chain: {chain_name}")
    return chain


def parse_model_family(family: str) -> ModelFamily:
    if family == "lstm":
        return "lstm"
    if family == "transformer":
        return "transformer"
    if family == "transformer_lstm":
        return "transformer_lstm"
    raise typer.BadParameter(f"Unknown model family: {family}")


def require_segment(segment: str) -> SegmentName:
    if segment == "history":
        return "history"
    if segment == "evaluation":
        return "evaluation"
    raise typer.BadParameter("segment must be 'history' or 'evaluation'")


def resolve_pull_segment(
    config: ExperimentConfig,
    chain: ChainConfig,
    segment: SegmentName,
) -> tuple[Path, TimestampRange]:
    output_dir = config.output_root / "raw" / chain.name / segment
    timestamps = history_range_for_chain(chain) if segment == "history" else evaluation_range()
    return output_dir, timestamps


def emit_raw_validation_report(
    config: ExperimentConfig,
    chain: ChainConfig,
    segment: SegmentName,
) -> bool:
    output_dir, timestamps = resolve_pull_segment(config, chain, segment)
    report = validate_raw_pull(
        output_dir,
        expected_chain_name=chain.name,
        expected_chain_id=chain.chain_id,
        expected_start_timestamp=timestamps.start,
        expected_end_timestamp=timestamps.end,
    )
    for line in format_raw_pull_validation_report(report):
        typer.echo(line)
    return report.status != "error"


@app.command("show-config")
def show_config(config_path: Path) -> None:
    """Print a quick summary of the experiment configuration."""
    config = ExperimentConfig.from_yaml(config_path)
    typer.echo(f"Output root: {config.output_root}")
    typer.echo(f"Max delays: {config.max_delay_seconds}")
    typer.echo(f"Lookback: {config.lookback_seconds}s")
    typer.echo(f"Target anchors: {config.target_anchor_count}")
    typer.echo(
        "Pull: "
        f"requests_per_second={config.pull.requests_per_second}, "
        f"max_concurrent_requests={config.pull.max_concurrent_requests}, "
        f"max_concurrent_chunks={config.pull.max_concurrent_chunks}"
    )
    typer.echo(
        "Simulation: "
        f"window_seconds={config.simulation.window_seconds}, "
        f"arrival_rate_per_second={config.simulation.arrival_rate_per_second}, "
        f"repetitions={config.simulation.repetitions}, "
        f"seed={config.simulation.seed}"
    )
    typer.echo("Chains:")
    for chain in config.chains:
        typer.echo(
            f"  - {chain.name}: chain_id={chain.chain_id}, "
            f"block_time={chain.block_time_seconds}s, "
            f"history_days={chain.history_days}"
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
    chain = require_chain(config, chain_name)
    client = RpcClient(resolve_rpc_url(chain.name))
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
    validate_on_success: bool = False,
) -> None:
    """Run cryo for one chain and one dataset segment."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = require_chain(config, chain_name)
    segment_name = require_segment(segment)
    if dry_run and validate_on_success:
        raise typer.BadParameter("Cannot use --validate-on-success with dry-run pulls")

    output_dir, timestamps = resolve_pull_segment(config, chain, segment_name)
    result = run_cryo(
        chain,
        config.pull,
        output_dir,
        timestamps,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if result.stdout:
        typer.echo(redact_sensitive_text(result.stdout.rstrip()))
    if result.stderr:
        typer.echo(redact_sensitive_text(result.stderr.rstrip()))
    if validate_on_success:
        if not emit_raw_validation_report(config, chain, segment_name):
            raise typer.Exit(code=1)


@app.command("validate-pull")
def validate_pull(config_path: Path, chain_name: str, segment: str) -> None:
    """Validate one completed raw block pull without mutating its files."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = require_chain(config, chain_name)
    segment_name = require_segment(segment)
    if not emit_raw_validation_report(config, chain, segment_name):
        raise typer.Exit(code=1)


@app.command("train")
def train(
    config_path: Path,
    history_block_path: Path,
    artifact_dir: Path,
    chain_name: str,
    family: str,
    max_delay_seconds: int,
    device: str = "auto",
) -> None:
    """Train one temporal model and write a canonical artifact directory."""
    config = ExperimentConfig.from_yaml(config_path)
    chain = require_chain(config, chain_name)
    model_config = ModelConfig(family=parse_model_family(family))

    result = run_training(
        history_block_path=history_block_path,
        chain=chain,
        max_delay_seconds=max_delay_seconds,
        lookback_seconds=config.lookback_seconds,
        target_anchor_count=config.target_anchor_count,
        model_config=model_config,
        training_config=replace(config.training, device=device),
        split_config=config.split,
    )
    manifest = build_training_artifact_manifest(
        result.prepared,
        chain=chain,
        max_delay_seconds=max_delay_seconds,
        lookback_seconds=config.lookback_seconds,
        target_anchor_count=config.target_anchor_count,
        model_config=model_config,
    )
    write_training_artifact(
        artifact_dir,
        manifest=manifest,
        model=result.model,
    )
    report = build_training_run_report(
        result,
        manifest=manifest,
        prepared=result.prepared,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        device_requested=device,
    )
    write_json_report(artifact_dir / TRAIN_REPORT_FILENAME, report)
    typer.echo(f"n_blocks_available={result.prepared.n_blocks_available}")
    typer.echo(f"n_blocks_used={result.prepared.n_blocks_used}")
    typer.echo(f"n_examples_total={result.prepared.n_examples_total}")
    typer.echo(f"lookback_steps={result.prepared.geometry.lookback_steps}")
    typer.echo(f"max_extra_wait_steps={result.prepared.geometry.max_extra_wait_steps}")
    typer.echo(f"action_count={result.prepared.geometry.action_count}")
    typer.echo(f"n_features={result.prepared.n_features}")
    typer.echo(f"train_examples={result.prepared.split_indices.train.shape[0]}")
    typer.echo(f"validation_examples={result.prepared.split_indices.validation.shape[0]}")
    typer.echo(f"test_examples={result.prepared.split_indices.test.shape[0]}")
    typer.echo(f"best_epoch={result.training_result.best_epoch}")
    typer.echo(f"test_loss={result.test_metrics.total_loss:.6f}")
    typer.echo(f"test_accuracy={result.test_metrics.accuracy:.4f}")
    typer.echo(
        f"test_profit_over_baseline={result.test_metrics.mean_profit_over_baseline:.6f}"
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
    config = ExperimentConfig.from_yaml(config_path)
    loaded_artifact = load_training_artifact(artifact_dir)
    geometry = derive_dataset_geometry(
        lookback_seconds=loaded_artifact.manifest.lookback_seconds,
        max_delay_seconds=loaded_artifact.manifest.max_delay_seconds,
        block_time_seconds=loaded_artifact.manifest.chain.block_time_seconds,
    )
    history_blocks = load_block_records(history_block_path)
    evaluation_blocks = load_block_records(evaluation_block_path)
    prepared = prepare_inference_dataset(
        history_blocks,
        evaluation_blocks,
        geometry=geometry,
        scaler=loaded_artifact.manifest.scaler,
    )
    predicted_offsets = predict_class_offsets(
        loaded_artifact.model,
        store=prepared.store,
        sample_indices=prepared.sample_indices,
        lookback_steps=prepared.geometry.lookback_steps,
        effective_batch_size=config.training.effective_batch_size,
        device=device,
    )
    simulation = run_temporal_simulation(
        prepared.store,
        predicted_offsets,
        sample_indices=prepared.sample_indices,
        window_seconds=config.simulation.window_seconds,
        arrival_rate_per_second=config.simulation.arrival_rate_per_second,
        repetitions=config.simulation.repetitions,
        seed=config.simulation.seed,
    )
    report = build_simulation_report(
        loaded_artifact,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        evaluation_block_path=evaluation_block_path,
        prepared=prepared,
        simulation=simulation,
        simulation_window_seconds=config.simulation.window_seconds,
        arrival_rate_per_second=config.simulation.arrival_rate_per_second,
        repetitions=config.simulation.repetitions,
    )
    write_json_report(artifact_dir / SIMULATION_REPORT_FILENAME, report)
    typer.echo(f"n_history_context_blocks={prepared.n_history_context_blocks}")
    typer.echo(f"n_evaluation_blocks={prepared.n_evaluation_blocks}")
    typer.echo(f"n_examples_total={prepared.n_examples_total}")
    typer.echo(f"simulation_profit_over_baseline={simulation.mean_profit_over_baseline:.6f}")
    typer.echo(f"simulation_cost_over_optimum={simulation.mean_cost_over_optimum:.6f}")
    typer.echo(
        f"simulation_baseline_cost_over_optimum={simulation.mean_baseline_cost_over_optimum:.6f}"
    )
    typer.echo(f"simulation_report_path={artifact_dir / SIMULATION_REPORT_FILENAME}")


if __name__ == "__main__":
    app()

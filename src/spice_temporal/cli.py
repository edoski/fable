"""Command-line interface for the SPICE temporal baseline project."""

from __future__ import annotations

from pathlib import Path

import typer

from spice_temporal.api import (
    BlockAcquireResult,
    BlockPullResult,
    _acquire_blocks,
    _enrich_blocks,
    _plan_block_pulls,
    _promote_block_pull,
    _pull_blocks,
    _stage_block_pull,
    _validate_block_pull,
    run_simulation_workflow,
    run_training_workflow,
)
from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from spice_temporal.config import BlockSegment, ChainName, ModelFamily
from spice_temporal.env import load_project_env
from spice_temporal.raw_validation import (
    RawPullValidationReport,
    format_raw_pull_validation_report,
)
from spice_temporal.rpc_providers import (
    RpcProviderName,
    redact_sensitive_text,
    resolve_rpc_provider,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
blocks_app = typer.Typer(no_args_is_help=True)
app.add_typer(blocks_app, name="blocks")


def _echo_validation_report(report: RawPullValidationReport) -> None:
    for line in format_raw_pull_validation_report(report):
        typer.echo(line)


def _echo_block_pull_result(
    result: BlockPullResult,
    *,
    rpc_provider: RpcProviderName | None,
    chain_name: ChainName,
) -> None:
    provider = None
    if result.process.stdout:
        load_project_env()
        provider = resolve_rpc_provider(rpc_provider, chains=(chain_name,))
        typer.echo(redact_sensitive_text(result.process.stdout.rstrip(), provider))
    if result.process.stderr:
        if provider is None:
            load_project_env()
            provider = resolve_rpc_provider(rpc_provider, chains=(chain_name,))
        typer.echo(redact_sensitive_text(result.process.stderr.rstrip(), provider))
    if result.source_manifest_path is not None:
        typer.echo(f"source_manifest_path={result.source_manifest_path}")
    if result.validation is not None:
        _echo_validation_report(result.validation)
        if result.validation.status == "error":
            raise typer.Exit(code=1)


def _echo_block_acquire_result(
    result: BlockAcquireResult,
    *,
    rpc_provider: RpcProviderName | None,
    pull_rpc_provider: RpcProviderName | None,
    chain_name: ChainName,
) -> None:
    _echo_block_pull_result(
        result.raw,
        rpc_provider=pull_rpc_provider or rpc_provider,
        chain_name=chain_name,
    )
    typer.echo(f"enriched_output_dir={result.enriched_output_dir}")
    typer.echo(f"enriched_files={result.enriched_file_count}")
    typer.echo(f"enriched_source_manifest_path={result.enriched_source_manifest_path}")


@blocks_app.command("plan")
def plan_blocks(
    config_path: Path,
    rpc_provider: RpcProviderName | None = None,
) -> None:
    """Render the cryo commands required for the baseline pull."""
    for plan in _plan_block_pulls(config_path, rpc_provider=rpc_provider):
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
    rpc_provider: RpcProviderName | None = None,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    """Add gas_limit to cryo block files using direct JSON-RPC lookups."""
    written = _enrich_blocks(
        config_path,
        chain_name,
        input_path,
        output_path,
        rpc_provider=rpc_provider,
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
    rpc_provider: RpcProviderName | None = None,
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
            rpc_provider=rpc_provider,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _echo_block_pull_result(result, rpc_provider=rpc_provider, chain_name=chain_name)


@blocks_app.command("stage")
def stage_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> None:
    """Run cryo for one chain and one dataset segment into provider-scoped staging."""
    try:
        result = _stage_block_pull(
            config_path,
            chain_name,
            segment,
            rpc_provider=rpc_provider,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _echo_block_pull_result(result, rpc_provider=rpc_provider, chain_name=chain_name)


@blocks_app.command("acquire")
def acquire_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    rpc_provider: RpcProviderName | None = None,
    pull_rpc_provider: RpcProviderName | None = None,
    enrich_rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = True,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    """Stage raw blocks and enrich them into a staged enriched dataset."""
    try:
        result = _acquire_blocks(
            config_path,
            chain_name,
            segment,
            rpc_provider=rpc_provider,
            pull_rpc_provider=pull_rpc_provider,
            enrich_rpc_provider=enrich_rpc_provider,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _echo_block_acquire_result(
        result,
        rpc_provider=rpc_provider,
        pull_rpc_provider=pull_rpc_provider,
        chain_name=chain_name,
    )


@blocks_app.command("validate")
def validate_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
) -> None:
    """Validate one completed raw block pull without mutating its files."""
    report = _validate_block_pull(config_path, chain_name, segment)
    _echo_validation_report(report)
    if report.status == "error":
        raise typer.Exit(code=1)


@blocks_app.command("promote")
def promote_blocks(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    source_path: Path,
    overwrite: bool = False,
    allow_warnings: bool = False,
) -> None:
    """Validate a staged raw dataset and promote it into the canonical baseline path."""
    try:
        result = _promote_block_pull(
            config_path,
            chain_name,
            segment,
            source_path,
            overwrite=overwrite,
            allow_warnings=allow_warnings,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"source_path={result.source_dir}")
    typer.echo(f"output_dir={result.output_dir}")
    typer.echo(f"source_manifest_path={result.source_manifest_path}")
    _echo_validation_report(result.validation)


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

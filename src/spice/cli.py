"""Command-line interface for SPICE."""

from __future__ import annotations

from pathlib import Path

import typer

from .acquisition.raw_validation import RawPullValidationReport, format_raw_pull_validation_report
from .acquisition.rpc_providers import RpcProviderName
from .api import (
    BlockAcquireResult,
    BlockPullResult,
    acquire_blocks,
    load_config,
    plan_block_pulls,
    promote_blocks,
    pull_blocks,
    run_simulation_workflow,
    run_training_workflow,
    stage_blocks,
    validate_blocks,
)
from .api import (
    enrich_blocks as run_enrich_blocks,
)
from .core.config import BlockSegment, ChainName, ModelFamily
from .core.console import RichReporter
from .core.constants import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME

app = typer.Typer(no_args_is_help=True, add_completion=False)
blocks_app = typer.Typer(no_args_is_help=True)
app.add_typer(blocks_app, name="blocks")


def _echo_validation_report(report: RawPullValidationReport) -> None:
    for line in format_raw_pull_validation_report(report):
        typer.echo(line)


def _echo_block_pull_result(result: BlockPullResult) -> None:
    typer.echo(f"output_dir={result.output_dir}")
    typer.echo(f"command={result.command}")
    typer.echo(f"completed_chunks={result.completed_chunks}")
    typer.echo(f"expected_chunks={result.expected_chunks}")
    if result.source_manifest_path is not None:
        typer.echo(f"source_manifest_path={result.source_manifest_path}")
    if result.validation is not None:
        _echo_validation_report(result.validation)
        if result.validation.status == "error":
            raise typer.Exit(code=1)


def _echo_block_acquire_result(result: BlockAcquireResult) -> None:
    _echo_block_pull_result(result.raw)
    typer.echo(f"enriched_output_dir={result.enriched_output_dir}")
    typer.echo(f"enriched_files={result.enriched_file_count}")
    typer.echo(f"enriched_source_manifest_path={result.enriched_source_manifest_path}")


@blocks_app.command("plan")
def plan_blocks(
    config_path: Path,
    rpc_provider: RpcProviderName | None = None,
) -> None:
    config = load_config(config_path)
    for plan in plan_block_pulls(config, rpc_provider=rpc_provider):
        typer.echo(f"[{plan.chain}]")
        typer.echo(f"history={plan.history_range.start}:{plan.history_range.end}")
        typer.echo(f"evaluation={plan.evaluation_range.start}:{plan.evaluation_range.end}")
        typer.echo(plan.command)
        typer.echo("")


@blocks_app.command("enrich")
def enrich_blocks_command(
    config_path: Path,
    chain_name: ChainName,
    input_path: Path,
    output_path: Path,
    rpc_provider: RpcProviderName | None = None,
    batch_size: int = 100,
    max_methods_per_second: float = 20.0,
) -> None:
    config = load_config(config_path)
    with RichReporter() as reporter:
        written = run_enrich_blocks(
            config,
            chain_name,
            input_path,
            output_path,
            rpc_provider=rpc_provider,
            batch_size=batch_size,
            max_methods_per_second=max_methods_per_second,
            reporter=reporter,
        )
    typer.echo(f"enriched_files={len(written)}")
    if written:
        typer.echo(f"first_output={written[0]}")


@blocks_app.command("pull")
def pull_blocks_command(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> None:
    config = load_config(config_path)
    with RichReporter() as reporter:
        result = pull_blocks(
            config,
            chain_name,
            segment,
            rpc_provider=rpc_provider,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
            reporter=reporter,
            config_path=config_path,
        )
    _echo_block_pull_result(result)


@blocks_app.command("stage")
def stage_blocks_command(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    rpc_provider: RpcProviderName | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    validate_on_success: bool = False,
) -> None:
    config = load_config(config_path)
    with RichReporter() as reporter:
        result = stage_blocks(
            config,
            chain_name,
            segment,
            rpc_provider=rpc_provider,
            dry_run=dry_run,
            overwrite=overwrite,
            validate_on_success=validate_on_success,
            reporter=reporter,
            config_path=config_path,
        )
    _echo_block_pull_result(result)


@blocks_app.command("acquire")
def acquire_blocks_command(
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
    config = load_config(config_path)
    with RichReporter() as reporter:
        result = acquire_blocks(
            config,
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
            reporter=reporter,
            config_path=config_path,
        )
    _echo_block_acquire_result(result)


@blocks_app.command("validate")
def validate_blocks_command(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
) -> None:
    report = validate_blocks(load_config(config_path), chain_name, segment)
    _echo_validation_report(report)
    if report.status == "error":
        raise typer.Exit(code=1)


@blocks_app.command("promote")
def promote_blocks_command(
    config_path: Path,
    chain_name: ChainName,
    segment: BlockSegment,
    source_path: Path,
    overwrite: bool = False,
    allow_warnings: bool = False,
) -> None:
    result = promote_blocks(
        load_config(config_path),
        chain_name,
        segment,
        source_path,
        overwrite=overwrite,
        allow_warnings=allow_warnings,
    )
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
    config = load_config(config_path)
    with RichReporter() as reporter:
        report = run_training_workflow(
            config,
            history_block_path,
            artifact_dir,
            chain_name,
            family,
            max_delay_seconds,
            device=device,
            reporter=reporter,
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
    report = run_simulation_workflow(
        load_config(config_path),
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

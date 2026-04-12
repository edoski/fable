"""Canonical block dataset acquisition workflow."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ..acquisition.datasets import (
    build_history_plan,
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..acquisition.metadata import (
    build_dataset_metadata,
    load_dataset_metadata,
    merge_providers,
    provider_metadata,
)
from ..acquisition.rpc import RpcController, Web3BlockClient, evaluation_range
from ..acquisition.windowing import required_history_block_count
from ..core.config import ExperimentConfig
from ..core.console import Reporter
from ..core.files import promote_paths_atomic
from ..core.json import write_json
from ..core.tracking import log_artifacts
from ._cli import load_cli_config
from ._shared import managed_workflow


def _chain_label(chain_name: str) -> str:
    return chain_name.replace("_", " ").title()


def _format_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _format_duration(start_timestamp: int, end_timestamp: int) -> str:
    remaining = max(0, end_timestamp - start_timestamp)
    units = (
        ("d", 24 * 60 * 60),
        ("h", 60 * 60),
        ("m", 60),
        ("s", 1),
    )
    parts: list[str] = []
    for suffix, size in units:
        if remaining < size and parts:
            continue
        value, remaining = divmod(remaining, size)
        if value > 0 or not parts:
            parts.append(f"{value}{suffix}")
        if len(parts) == 2:
            break
    return " ".join(parts)


def _format_count(value: int, singular: str, plural: str | None = None) -> str:
    unit = singular if value == 1 else (plural or f"{singular}s")
    return f"{value:,} {unit}"


def _planned_window_rows(
    *,
    start_timestamp: int,
    end_timestamp: int,
    expected_rows: int,
    expected_files: int,
) -> list[tuple[str, str]]:
    return [
        ("window", f"{_format_timestamp(start_timestamp)} -> {_format_timestamp(end_timestamp)}"),
        ("duration", _format_duration(start_timestamp, end_timestamp)),
        (
            "planned",
            f"{_format_count(expected_rows, 'block')} in {_format_count(expected_files, 'file')}",
        ),
    ]


def _final_window_rows(
    *,
    row_count: int,
    file_count: int,
    reused: bool,
) -> list[tuple[str, str]]:
    result = _format_count(row_count, "block")
    if reused:
        result = f"{result} reused"
    else:
        result = f"{result} in {_format_count(file_count, 'file')}"
    return [("result", result)]


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    asyncio.run(_run_async(config, reporter=reporter))


async def _run_async(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    history_dir = config.paths.history_dir
    evaluation_dir = config.paths.evaluation_dir
    metadata_path = config.paths.dataset_metadata_path
    rpc_controller = RpcController.from_config(config.acquisition)

    required_history_blocks = required_history_block_count(config)
    evaluation_window = evaluation_range(
        config.evaluation_window_start_timestamp,
        config.evaluation_window_end_timestamp,
    )
    chain_label = _chain_label(config.chain.name.value)

    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name.value}-{config.provider.name.value}",
        reporter=reporter,
    ) as session:
        block_client = Web3BlockClient(config.provider, config.chain)
        try:
            evaluation_plan = await block_client.plan_window(
                evaluation_window,
                chunk_size=config.acquisition.chunk_size,
            )
            history_plan = await build_history_plan(
                config=config,
                block_client=block_client,
                required_history_blocks=required_history_blocks,
            )

            if config.acquisition.dry_run:
                session.runtime.log_sectioned_summary(
                    "acquire dry run",
                    [
                        (
                            "dataset",
                            [
                                ("id", config.dataset.id),
                                ("chain", chain_label),
                                ("evaluation date", str(config.evaluation.date)),
                                (
                                    "required history",
                                    _format_count(required_history_blocks, "block"),
                                ),
                            ],
                        ),
                        (
                            "history",
                            _planned_window_rows(
                                start_timestamp=history_plan.window.start,
                                end_timestamp=history_plan.window.end,
                                expected_rows=history_plan.expected_rows,
                                expected_files=history_plan.expected_files,
                            ),
                        ),
                        (
                            "evaluation",
                            _planned_window_rows(
                                start_timestamp=evaluation_window.start,
                                end_timestamp=evaluation_window.end,
                                expected_rows=evaluation_plan.expected_rows,
                                expected_files=evaluation_plan.expected_files,
                            ),
                        ),
                    ],
                )
                return

            existing_metadata = load_dataset_metadata(metadata_path)
            try:
                config.paths.dataset_root.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(
                    dir=config.paths.dataset_root.parent,
                    prefix=f".{config.dataset.id}.acquire.",
                ) as temp_root_name:
                    temp_root = Path(temp_root_name)
                    history_result = await ensure_history_dataset(
                        config=config,
                        block_client=block_client,
                        output_dir=history_dir,
                        working_dir=temp_root,
                        history_plan=history_plan,
                        required_history_blocks=required_history_blocks,
                        rpc_controller=rpc_controller,
                        reporter=session.reporter,
                    )
                    evaluation_result = await ensure_evaluation_dataset(
                        config=config,
                        block_client=block_client,
                        output_dir=evaluation_dir,
                        working_dir=temp_root,
                        evaluation_plan=evaluation_plan,
                        rpc_controller=rpc_controller,
                        reporter=session.reporter,
                    )
                    providers = list(existing_metadata.providers) if existing_metadata else []
                    current_provider = provider_metadata(config)
                    if (
                        not providers
                        or history_result.pulled_blocks
                        or evaluation_result.pulled_blocks
                    ):
                        providers = merge_providers(providers, current_provider)
                    metadata = build_dataset_metadata(
                        config=config,
                        history_dir=history_dir,
                        evaluation_dir=evaluation_dir,
                        history_request_start_timestamp=history_plan.window.start,
                        history_request_end_timestamp=history_plan.window.end,
                        evaluation_request_start_timestamp=evaluation_window.start,
                        evaluation_request_end_timestamp=evaluation_window.end,
                        providers=providers,
                        history_validation=history_result.validation,
                        evaluation_validation=evaluation_result.validation,
                        acquisition_runtime=rpc_controller.snapshot(),
                    )
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_task = session.reporter.start_task("write dataset metadata")
                    metadata_tmp_path = temp_root / ".spice" / "metadata.json"
                    write_json(metadata_tmp_path, metadata)
                    promotions: list[tuple[Path, Path]] = []
                    if history_result.promote_dir is not None:
                        promotions.append((history_dir, history_result.promote_dir))
                    if evaluation_result.promote_dir is not None:
                        promotions.append((evaluation_dir, evaluation_result.promote_dir))
                    promotions.append((metadata_path, metadata_tmp_path))
                    promote_paths_atomic(promotions)
                    session.reporter.finish_task(
                        metadata_task,
                        message=str(metadata_path),
                        silent=True,
                    )
            except KeyboardInterrupt:
                session.reporter.log(
                    "acquire interrupted; temporary outputs removed, canonical dataset preserved",
                    level="warning",
                )
                raise
            except Exception:
                session.reporter.log(
                    "acquire failed; temporary outputs removed, canonical dataset preserved",
                    level="warning",
                )
                raise
            session.runtime.log_sectioned_summary(
                "acquisition summary",
                [
                    (
                        "dataset",
                        [
                            ("id", config.dataset.id),
                            ("chain", chain_label),
                        ],
                    ),
                    (
                        "history",
                        _final_window_rows(
                            row_count=history_result.validation.row_count,
                            file_count=history_result.file_count,
                            reused=history_result.reused,
                        ),
                    ),
                    (
                        "evaluation",
                        _final_window_rows(
                            row_count=evaluation_result.validation.row_count,
                            file_count=evaluation_result.file_count,
                            reused=evaluation_result.reused,
                        ),
                    ),
                ],
            )
            if session.tracking_enabled:
                import mlflow

                mlflow.log_metrics(
                    {
                        "history_files": float(history_result.file_count),
                        "evaluation_files": float(evaluation_result.file_count),
                        "history_gap_count": float(history_result.validation.gap_count),
                        "evaluation_gap_count": float(evaluation_result.validation.gap_count),
                        "rpc_final_batch_size": float(rpc_controller.current_batch_size),
                        "rpc_final_concurrency": float(rpc_controller.current_concurrency),
                    }
                )
                log_artifacts([metadata_path])
        finally:
            await block_client.close()


def main(argv: list[str] | None = None) -> None:
    run(load_cli_config("acquire", prog="spice-acquire", argv=argv))


if __name__ == "__main__":
    main()

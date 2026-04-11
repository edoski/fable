"""Hydra entrypoint for canonical block dataset acquisition."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import hydra
from omegaconf import DictConfig

from ..acquisition.datasets import (
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..acquisition.metadata import (
    build_dataset_metadata,
    check_existing_dataset_metadata,
)
from ..acquisition.rpc import RpcController, Web3BlockClient, evaluation_range
from ..acquisition.windowing import (
    history_range_from_metadata,
    required_history_block_count,
)
from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter
from ..core.json import write_json
from ..core.tracking import log_artifacts
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
    expected_files: int | None,
    reused: bool,
) -> list[tuple[str, str]]:
    result = _format_count(row_count, "block")
    if expected_files is not None:
        result = f"{result} in {_format_count(expected_files, 'file')}"
    elif reused:
        result = f"{result} reused"
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
        config.dataset.window.start_timestamp,
        config.dataset.window.end_timestamp,
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
            existing_metadata = check_existing_dataset_metadata(
                config=config,
                metadata_path=metadata_path,
                overwrite=config.acquisition.overwrite,
            )
            if existing_metadata is not None and not config.acquisition.overwrite:
                history_plan = await block_client.plan_window(
                    history_range_from_metadata(existing_metadata),
                    chunk_size=config.acquisition.chunk_size,
                )
            else:
                history_plan = await block_client.plan_history_window(
                    end_timestamp=config.dataset.window.start_timestamp,
                    required_history_blocks=required_history_blocks,
                    chunk_size=config.acquisition.chunk_size,
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

            history_result, history_validation, resolved_history_plan = await ensure_history_dataset(
                config=config,
                block_client=block_client,
                output_dir=history_dir,
                history_plan=history_plan,
                required_history_blocks=required_history_blocks,
                rpc_controller=rpc_controller,
                reporter=session.reporter,
            )
            evaluation_result, evaluation_validation = await ensure_evaluation_dataset(
                config=config,
                block_client=block_client,
                output_dir=evaluation_dir,
                evaluation_plan=evaluation_plan,
                rpc_controller=rpc_controller,
                reporter=session.reporter,
            )
            metadata = build_dataset_metadata(
                config=config,
                history_dir=history_dir,
                evaluation_dir=evaluation_dir,
                history_window_start=resolved_history_plan.window.start,
                history_window_end=resolved_history_plan.window.end,
                evaluation_window_start=evaluation_window.start,
                evaluation_window_end=evaluation_window.end,
                history_validation=history_validation,
                evaluation_validation=evaluation_validation,
                acquisition_runtime=rpc_controller.snapshot(),
            )
            metadata_task = session.reporter.start_task("write dataset metadata")
            write_json(metadata_path, metadata)
            session.reporter.finish_task(
                metadata_task,
                message=str(metadata_path),
                silent=True,
            )
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
                            row_count=history_validation.row_count,
                            expected_files=(
                                None if history_result is None else history_result.expected_files
                            ),
                            reused=history_result is None,
                        ),
                    ),
                    (
                        "evaluation",
                        _final_window_rows(
                            row_count=evaluation_validation.row_count,
                            expected_files=(
                                None
                                if evaluation_result is None
                                else evaluation_result.expected_files
                            ),
                            reused=evaluation_result is None,
                        ),
                    ),
                ],
            )
            if session.tracking_enabled:
                import mlflow

                mlflow.log_metrics(
                    {
                        "history_files": float(
                            0 if history_result is None else history_result.expected_files
                        ),
                        "evaluation_files": float(
                            0 if evaluation_result is None else evaluation_result.expected_files
                        ),
                        "history_gap_count": float(history_validation.gap_count),
                        "evaluation_gap_count": float(evaluation_validation.gap_count),
                        "rpc_final_batch_size": float(rpc_controller.current_batch_size),
                        "rpc_final_concurrency": float(rpc_controller.current_concurrency),
                    }
                )
                log_artifacts([metadata_path])
        finally:
            await block_client.close()


@hydra.main(version_base=None, config_path="../conf", config_name="acquire")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="acquire"))


if __name__ == "__main__":
    main()

"""Hydra entrypoint for canonical block dataset acquisition."""

from __future__ import annotations

from pathlib import Path

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
from ..acquisition.rpc import Web3BlockClient, evaluation_range
from ..acquisition.windowing import (
    history_range_from_metadata,
    required_history_block_count,
)
from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter
from ..core.json import write_json
from ..core.tracking import log_artifacts
from ._shared import managed_workflow


def _window_summary_rows(
    label: str,
    *,
    start_timestamp: int,
    end_timestamp: int,
    block_start: int,
    block_end: int,
    expected_rows: int,
    expected_files: int,
) -> list[tuple[str, str]]:
    return [
        (f"{label} window", f"{start_timestamp}..{end_timestamp}"),
        (f"{label} blocks", f"{block_start}..{block_end}"),
        (f"{label} expected", f"{expected_rows} rows in {expected_files} files"),
    ]


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    history_dir = Path(config.paths.history_dir)
    evaluation_dir = Path(config.paths.evaluation_dir)
    metadata_path = Path(config.paths.dataset_metadata_path)

    required_history_blocks = required_history_block_count(config)
    evaluation_window = evaluation_range(
        config.dataset.window.start_timestamp,
        config.dataset.window.end_timestamp,
    )
    block_client = Web3BlockClient(config.provider, config.chain)
    evaluation_plan = block_client.plan_window(
        evaluation_window,
        chunk_size=config.acquisition.chunk_size,
    )
    existing_metadata = check_existing_dataset_metadata(
        config=config,
        metadata_path=metadata_path,
        overwrite=config.acquisition.overwrite,
    )
    if existing_metadata is not None and not config.acquisition.overwrite:
        history_plan = block_client.plan_window(
            history_range_from_metadata(existing_metadata),
            chunk_size=config.acquisition.chunk_size,
        )
    else:
        history_plan = block_client.plan_history_window(
            end_timestamp=config.dataset.window.start_timestamp,
            required_history_blocks=required_history_blocks,
            chunk_size=config.acquisition.chunk_size,
        )

    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name.value}-{config.provider.name.value}",
        reporter=reporter,
    ) as session:
        if config.acquisition.dry_run:
            session.runtime.log_summary(
                "acquire dry run",
                [("dataset", config.dataset.id)]
                + _window_summary_rows(
                    "history",
                    start_timestamp=history_plan.window.start,
                    end_timestamp=history_plan.window.end,
                    block_start=history_plan.block_range.start,
                    block_end=history_plan.block_range.end,
                    expected_rows=history_plan.expected_rows,
                    expected_files=history_plan.expected_files,
                )
                + _window_summary_rows(
                    "evaluation",
                    start_timestamp=evaluation_window.start,
                    end_timestamp=evaluation_window.end,
                    block_start=evaluation_plan.block_range.start,
                    block_end=evaluation_plan.block_range.end,
                    expected_rows=evaluation_plan.expected_rows,
                    expected_files=evaluation_plan.expected_files,
                ),
            )
            return

        history_result, history_validation, resolved_history_plan = ensure_history_dataset(
            config=config,
            block_client=block_client,
            output_dir=history_dir,
            history_plan=history_plan,
            required_history_blocks=required_history_blocks,
            reporter=session.reporter,
        )
        evaluation_result, evaluation_validation = ensure_evaluation_dataset(
            config=config,
            block_client=block_client,
            output_dir=evaluation_dir,
            evaluation_plan=evaluation_plan,
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
        )
        metadata_task = session.reporter.start_task("write dataset metadata")
        write_json(metadata_path, metadata)
        session.reporter.finish_task(metadata_task, message=str(metadata_path))
        session.runtime.log_summary(
            "acquisition summary",
            [
                ("dataset", config.dataset.id),
                (
                    "history",
                    f"{history_validation.row_count} rows, "
                    f"{0 if history_result is None else history_result.expected_files} files, "
                    f"validation={history_validation.status}",
                ),
                (
                    "evaluation",
                    f"{evaluation_validation.row_count} rows, "
                    f"{0 if evaluation_result is None else evaluation_result.expected_files} "
                    f"files, validation={evaluation_validation.status}",
                ),
                ("required history blocks", str(required_history_blocks)),
                ("metadata", str(metadata_path)),
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
                }
            )
            log_artifacts([metadata_path])


@hydra.main(version_base=None, config_path="../conf", config_name="acquire")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="acquire"))


if __name__ == "__main__":
    main()

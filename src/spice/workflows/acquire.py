"""Hydra entrypoint for dataset acquisition and enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from ..acquisition.cryo import evaluation_range
from ..acquisition.datasets import (
    ensure_enriched_dataset,
    ensure_evaluation_raw_dataset,
    ensure_history_raw_dataset,
    run_raw_pull,
)
from ..acquisition.metadata import (
    build_dataset_metadata,
    check_existing_dataset_metadata,
)
from ..acquisition.rpc import Web3BlockClient
from ..acquisition.windowing import (
    history_range_from_metadata,
    initial_history_range,
    required_history_block_count,
)
from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter, RichReporter
from ..core.json import write_json
from ..core.tracking import log_artifacts
from ._shared import managed_workflow


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    raw_history_dir = Path(config.paths.raw_history_dir)
    raw_evaluation_dir = Path(config.paths.raw_evaluation_dir)
    enriched_history_dir = Path(config.paths.enriched_history_dir)
    enriched_evaluation_dir = Path(config.paths.enriched_evaluation_dir)
    metadata_path = Path(config.paths.dataset_metadata_path)

    required_history_blocks = required_history_block_count(config)
    history_window = initial_history_range(
        config,
        required_history_blocks=required_history_blocks,
    )
    evaluation_window = evaluation_range(
        config.dataset.window.start_timestamp,
        config.dataset.window.end_timestamp,
    )
    existing_metadata = check_existing_dataset_metadata(
        config=config,
        metadata_path=metadata_path,
        overwrite=config.acquisition.overwrite,
    )
    if existing_metadata is not None and not config.acquisition.overwrite:
        history_window = history_range_from_metadata(existing_metadata)

    block_client = Web3BlockClient(config.provider, config.chain)
    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name.value}-{config.provider.name.value}",
        reporter=reporter,
        default_reporter_factory=RichReporter,
    ) as session:
        if config.acquisition.dry_run:
            history_result = run_raw_pull(
                config,
                output_dir=raw_history_dir,
                window=history_window,
                reporter=session.reporter,
                overwrite=config.acquisition.overwrite,
                dry_run=True,
            )
            evaluation_result = run_raw_pull(
                config,
                output_dir=raw_evaluation_dir,
                window=evaluation_window,
                reporter=session.reporter,
                overwrite=config.acquisition.overwrite,
                dry_run=True,
            )
            session.reporter.log(
                json.dumps(
                    {
                        "history_completed_chunks": history_result.completed_chunks,
                        "evaluation_completed_chunks": evaluation_result.completed_chunks,
                        "dataset_id": config.dataset.id,
                        "history_window": {
                            "start_timestamp": history_window.start,
                            "end_timestamp": history_window.end,
                        },
                        "evaluation_window": {
                            "start_timestamp": evaluation_window.start,
                            "end_timestamp": evaluation_window.end,
                        },
                        "history_validation": "dry_run",
                        "evaluation_validation": "dry_run",
                    }
                )
            )
            return

        history_result, history_validation, history_window = ensure_history_raw_dataset(
            config=config,
            output_dir=raw_history_dir,
            history_window=history_window,
            required_history_blocks=required_history_blocks,
            reporter=session.reporter,
        )
        evaluation_result, evaluation_validation = ensure_evaluation_raw_dataset(
            config=config,
            output_dir=raw_evaluation_dir,
            evaluation_window=evaluation_window,
            reporter=session.reporter,
        )
        history_enriched = ensure_enriched_dataset(
            input_dir=raw_history_dir,
            output_dir=enriched_history_dir,
            expected_chain_id=config.chain.chain_id,
            expected_start_timestamp=history_window.start,
            expected_end_timestamp=history_window.end,
            overwrite=config.acquisition.overwrite or history_result is not None,
            fetch_gas_limits=block_client.get_block_gas_limits,
            batch_size=config.acquisition.enrich_batch_size,
            max_methods_per_second=config.acquisition.max_methods_per_second,
            reporter=session.reporter,
        )
        evaluation_enriched = ensure_enriched_dataset(
            input_dir=raw_evaluation_dir,
            output_dir=enriched_evaluation_dir,
            expected_chain_id=config.chain.chain_id,
            expected_start_timestamp=evaluation_window.start,
            expected_end_timestamp=evaluation_window.end,
            overwrite=config.acquisition.overwrite or evaluation_result is not None,
            fetch_gas_limits=block_client.get_block_gas_limits,
            batch_size=config.acquisition.enrich_batch_size,
            max_methods_per_second=config.acquisition.max_methods_per_second,
            reporter=session.reporter,
        )
        metadata = build_dataset_metadata(
            config=config,
            raw_history_dir=raw_history_dir,
            raw_evaluation_dir=raw_evaluation_dir,
            enriched_history_dir=enriched_history_dir,
            enriched_evaluation_dir=enriched_evaluation_dir,
            history_window_start=history_window.start,
            history_window_end=history_window.end,
            evaluation_window_start=evaluation_window.start,
            evaluation_window_end=evaluation_window.end,
            history_validation=history_validation,
            evaluation_validation=evaluation_validation,
            history_enriched=history_enriched,
            evaluation_enriched=evaluation_enriched,
        )
        write_json(metadata_path, metadata)
        session.reporter.log(
            json.dumps(
                {
                    "history_completed_chunks": (
                        0 if history_result is None else history_result.completed_chunks
                    ),
                    "evaluation_completed_chunks": (
                        0 if evaluation_result is None else evaluation_result.completed_chunks
                    ),
                    "history_validation": history_validation.status,
                    "evaluation_validation": evaluation_validation.status,
                    "history_enriched": history_enriched.status,
                    "evaluation_enriched": evaluation_enriched.status,
                    "history_rows": history_validation.row_count,
                    "required_history_blocks": required_history_blocks,
                }
            )
        )
        if session.tracking_enabled:
            mlflow.log_metrics(
                {
                    "history_completed_chunks": float(
                        0 if history_result is None else history_result.completed_chunks
                    ),
                    "evaluation_completed_chunks": float(
                        0 if evaluation_result is None else evaluation_result.completed_chunks
                    ),
                    "history_gap_count": float(history_validation.gap_count),
                    "evaluation_gap_count": float(evaluation_validation.gap_count),
                    "history_overlap_count": float(history_validation.overlap_count),
                    "evaluation_overlap_count": float(evaluation_validation.overlap_count),
                }
            )
            log_artifacts([metadata_path])


@hydra.main(version_base=None, config_path="../conf", config_name="acquire")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="acquire"))


if __name__ == "__main__":
    main()

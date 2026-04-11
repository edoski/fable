"""Hydra entrypoint for training runs."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter, RichReporter
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from ..core.tracking import configure_mlflow, log_artifacts, log_config, log_epoch_history
from ..modeling.artifacts import build_training_artifact_manifest, write_training_artifact
from ..modeling.pipeline import run_training
from ..modeling.reporting import build_training_run_report, write_json_report
from ._shared import (
    apply_best_tuning_params,
    build_training_spec,
    epoch_metrics_to_dict,
    start_run_if_enabled,
)


def _clean_training_outputs(config: ExperimentConfig) -> None:
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    for path in (
        Path(config.paths.artifact_root) / ARTIFACT_MANIFEST_FILENAME,
        Path(config.paths.artifact_root) / MODEL_STATE_FILENAME,
        Path(config.paths.train_report_path),
        Path(config.paths.simulation_report_path),
    ):
        if path.exists():
            path.unlink()


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    if config.tuning.apply_best_params:
        config = apply_best_tuning_params(config)
    spec = build_training_spec(config)
    artifact_dir = Path(config.paths.artifact_root)
    history_block_path = Path(config.paths.enriched_history_dir)
    if config.tracking.enabled:
        configure_mlflow(config)

    active_reporter = reporter or RichReporter()
    run_context = start_run_if_enabled(
        config,
        run_name=f"train-{config.chain.name.value}-{config.model.family.value}-{config.max_delay_seconds}s",
    )
    try:
        if run_context is not None:
            run_context.__enter__()
            log_config(config)
            mlflow.set_tags(config.tracking.tags)

        _clean_training_outputs(config)
        result = run_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            reporter=active_reporter,
        )
        manifest = build_training_artifact_manifest(result.prepared, spec=spec)
        write_training_artifact(
            artifact_dir,
            manifest=manifest,
            model=result.model,
        )
        report = build_training_run_report(
            result,
            target_anchor_count=config.target_anchor_count,
            max_delay_seconds=config.max_delay_seconds,
            lookback_seconds=config.lookback_seconds,
            chain_name=config.chain.name.value,
            dataset_id=config.dataset.id,
            family=config.model.family.value,
            block_time_seconds=config.chain.block_time_seconds,
            manifest=manifest,
            prepared=result.prepared,
            artifact_dir=artifact_dir,
            history_block_path=history_block_path,
            device_requested=config.training.device,
        )
        report_path = Path(config.paths.train_report_path)
        write_json_report(report_path, report)

        if config.tracking.enabled:
            mlflow.log_metrics(
                {
                    "test_loss": report.test_metrics.total_loss,
                    "test_accuracy": report.test_metrics.accuracy,
                    "test_cost_over_optimum": report.test_metrics.mean_cost_over_optimum,
                    "test_profit_over_baseline": report.test_metrics.mean_profit_over_baseline,
                    "best_epoch": float(result.training_result.best_epoch),
                }
            )
            log_epoch_history(
                prefix="train",
                metrics_history=[
                    epoch_metrics_to_dict(item) for item in result.training_result.train_history
                ],
            )
            log_epoch_history(
                prefix="validation",
                metrics_history=[
                    epoch_metrics_to_dict(item)
                    for item in result.training_result.validation_history
                ],
            )
            log_artifacts(
                [
                    artifact_dir / ARTIFACT_MANIFEST_FILENAME,
                    artifact_dir / MODEL_STATE_FILENAME,
                    report_path,
                ]
            )
            if result.training_result.best_checkpoint_path is not None:
                log_artifacts([result.training_result.best_checkpoint_path])
    finally:
        if run_context is not None:
            run_context.__exit__(None, None, None)
        if reporter is None:
            active_reporter.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="train"))


if __name__ == "__main__":
    main()

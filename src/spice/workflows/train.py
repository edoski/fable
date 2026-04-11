"""Hydra entrypoint for training runs."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from ..core.tracking import log_artifacts, log_epoch_history
from ..modeling.execution import run_persisted_training
from ._shared import (
    apply_best_tuning_params,
    build_training_spec,
    epoch_metrics_to_dict,
    managed_workflow,
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
    report_path = Path(config.paths.train_report_path)
    history_block_path = Path(config.paths.enriched_history_dir)
    with managed_workflow(
        config,
        run_name=(
            "train-"
            f"{config.chain.name.value}-{config.model.family.value}-"
            f"{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        _clean_training_outputs(config)
        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            report_path=report_path,
            reporter=session.reporter,
        )
        if session.tracking_enabled:
            import mlflow

            mlflow.log_metrics(
                {
                    "test_loss": persisted.report.test_metrics.total_loss,
                    "test_accuracy": persisted.report.test_metrics.accuracy,
                    "test_cost_over_optimum": persisted.report.test_metrics.mean_cost_over_optimum,
                    "test_profit_over_baseline": (
                        persisted.report.test_metrics.mean_profit_over_baseline
                    ),
                    "best_epoch": float(persisted.training_run.training_result.best_epoch),
                }
            )
            log_epoch_history(
                prefix="train",
                metrics_history=[
                    epoch_metrics_to_dict(item)
                    for item in persisted.training_run.training_result.train_history
                ],
            )
            log_epoch_history(
                prefix="validation",
                metrics_history=[
                    epoch_metrics_to_dict(item)
                    for item in persisted.training_run.training_result.validation_history
                ],
            )
            log_artifacts(persisted.artifact_paths)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="train"))


if __name__ == "__main__":
    main()

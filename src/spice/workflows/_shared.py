"""Shared helpers for Hydra workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow

from ..core.config import ExperimentConfig
from ..modeling.lightning_module import EpochMetrics
from ..modeling.pipeline import TrainingSpec


def build_training_spec(config: ExperimentConfig) -> TrainingSpec:
    return TrainingSpec(
        chain=config.chain,
        model=config.model,
        max_delay_seconds=config.max_delay_seconds,
        lookback_seconds=config.lookback_seconds,
        target_anchor_count=config.target_anchor_count,
        split=config.split,
        training=config.training,
    )


def epoch_metrics_to_dict(metrics: EpochMetrics) -> dict[str, float]:
    return {
        "loss": metrics.total_loss,
        "accuracy": metrics.accuracy,
        "cost_over_optimum": metrics.mean_cost_over_optimum,
        "profit_over_baseline": metrics.mean_profit_over_baseline,
    }


def start_run_if_enabled(
    config: ExperimentConfig,
    *,
    run_name: str,
    nested: bool = False,
):
    if not config.tracking.enabled:
        return None
    return mlflow.start_run(run_name=run_name, nested=nested)


def trial_artifact_dir(config: ExperimentConfig, trial_number: int) -> Path:
    return Path(config.paths.tuning_root) / "trials" / f"trial-{trial_number:03d}"


def clone_config(config: ExperimentConfig) -> ExperimentConfig:
    return deepcopy(config)


def set_nested_attr(config: ExperimentConfig, dotted_path: str, value: Any) -> None:
    current: Any = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)

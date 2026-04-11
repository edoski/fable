"""Shared helpers for Hydra workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from ..core.config import ExperimentConfig, revalidate_config
from ..core.console import NullReporter, Reporter
from ..core.tracking import configure_mlflow, log_config
from ..modeling.evaluation import EpochMetrics
from ..modeling.pipeline import TrainingSpec


def build_training_spec(config: ExperimentConfig) -> TrainingSpec:
    return TrainingSpec(
        chain=config.chain,
        dataset_id=config.dataset.id,
        model=config.model,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        anchor_count=config.dataset.sampling.anchor_count,
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


@dataclass(slots=True)
class WorkflowSession:
    reporter: Reporter
    tracking_enabled: bool


@contextmanager
def managed_workflow(
    config: ExperimentConfig,
    *,
    run_name: str,
    reporter: Reporter | None = None,
    default_reporter_factory: Callable[[], Reporter] = NullReporter,
    nested: bool = False,
) -> Iterator[WorkflowSession]:
    active_reporter = reporter or default_reporter_factory()
    owns_reporter = reporter is None
    try:
        if config.tracking.enabled:
            configure_mlflow(config)
            with mlflow.start_run(run_name=run_name, nested=nested):
                log_config(config)
                mlflow.set_tags(config.tracking.tags)
                yield WorkflowSession(
                    reporter=active_reporter,
                    tracking_enabled=True,
                )
        else:
            yield WorkflowSession(
                reporter=active_reporter,
                tracking_enabled=False,
            )
    finally:
        if owns_reporter:
            active_reporter.close()


def trial_artifact_dir(config: ExperimentConfig, trial_number: int) -> Path:
    return Path(config.paths.tuning_root) / "trials" / f"trial-{trial_number:03d}"


def apply_best_tuning_params(config: ExperimentConfig) -> ExperimentConfig:
    tuned_config = clone_config(config)
    path = Path(tuned_config.paths.tuning_best_params_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(
            f"Best tuning params are required but missing: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("kind") != "tuning_best_params":
        raise ValueError(f"Invalid best tuning params file: {path}")
    params = payload.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Best tuning params file contains no params: {path}")
    for key, value in params.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid tuned parameter key in {path}: {key!r}")
        set_nested_attr(tuned_config, key, value)
    return revalidate_config(tuned_config)


def clone_config(config: ExperimentConfig) -> ExperimentConfig:
    return deepcopy(config)


def set_nested_attr(config: ExperimentConfig, dotted_path: str, value: Any) -> None:
    current: Any = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)

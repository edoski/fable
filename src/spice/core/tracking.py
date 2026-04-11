"""MLflow helpers for workflow runs."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from .config import ExperimentConfig, config_to_dict
from .json import JsonObject


def resolve_tracking_uri(config: ExperimentConfig) -> str:
    if config.tracking.tracking_uri:
        return config.tracking.tracking_uri
    database_path = config.paths.mlruns_dir.resolve() / "mlflow.db"
    return f"sqlite:///{database_path}"


def configure_mlflow(config: ExperimentConfig) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)
    tracking_root = config.paths.mlruns_dir.resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    if config.tracking.tracking_uri:
        mlflow.set_experiment(config.tracking.experiment_name)
        return

    artifact_root = tracking_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is None:
        client.create_experiment(
            config.tracking.experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    mlflow.set_experiment(config.tracking.experiment_name)


def _flatten_dict(payload: JsonObject, *, prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            flattened[full_key] = ",".join(str(item) for item in value)
        else:
            flattened[full_key] = str(value)
    return flattened


def log_config(config: ExperimentConfig) -> None:
    import mlflow

    flattened = _flatten_dict(config_to_dict(config))
    for key, value in flattened.items():
        mlflow.log_param(key, value)


def log_epoch_history(
    *,
    prefix: str,
    metrics_history: Iterable[dict[str, float]],
) -> None:
    import mlflow

    for step, metrics in enumerate(metrics_history, start=1):
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{prefix}.{metric_name}", value, step=step)


def log_artifacts(paths: Iterable[Path]) -> None:
    import mlflow

    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=path.parent.name)
        elif path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=path.name)

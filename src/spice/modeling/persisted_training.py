"""Persisted training orchestration and artifact/state writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..metrics import MetricSet
from ..storage.artifact import write_training_state
from .artifacts import (
    TrainingArtifactManifest,
    build_training_artifact_manifest,
    load_training_artifact,
    persist_training_artifact,
)
from .dataset_builders import PreparedTrainingDataset
from .pipeline import TrainingSpec, run_training
from .results import (
    LoadedTrainingSummary,
    build_training_runtime_summary,
    iter_epoch_records,
)
from .training_run import TrainingRunResult
from .training_runner import (
    EarlyStopCallback,
    EpochEndCallback,
    TrainingMetricEvaluationSpec,
    evaluate_training_metrics,
)


@dataclass(slots=True)
class PersistedTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    summary: LoadedTrainingSummary
    artifact_dir: Path


def _evaluate_split_metrics(
    training_run: TrainingRunResult,
    *,
    spec: TrainingSpec,
    model,
) -> tuple[MetricSet, MetricSet]:
    prepared = training_run.prepared
    best_validation_metrics = evaluate_training_metrics(
        TrainingMetricEvaluationSpec(
            model=model,
            model_config=spec.model,
            prediction_contract=spec.prediction_contract,
            representation_contract=spec.representation_contract,
            prepared=prepared,
            samples=prepared.samples.validation,
            prediction_training_state=training_run.prediction_training_state,
            training_config=spec.training,
        )
    )
    test_metrics = evaluate_training_metrics(
        TrainingMetricEvaluationSpec(
            model=model,
            model_config=spec.model,
            prediction_contract=spec.prediction_contract,
            representation_contract=spec.representation_contract,
            prepared=prepared,
            samples=prepared.samples.test,
            prediction_training_state=training_run.prediction_training_state,
            training_config=spec.training,
        )
    )
    return best_validation_metrics, test_metrics


def run_persisted_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    persist_artifact: bool = True,
    on_prepare_complete: Callable[[PreparedTrainingDataset], None] | None = None,
    on_fit_start: Callable[[], None] | None = None,
    on_epoch_end: EpochEndCallback | None = None,
    on_early_stop: EarlyStopCallback | None = None,
) -> PersistedTrainingRun:
    training_run = run_training(
        history_block_path,
        spec=spec,
        on_prepare_complete=on_prepare_complete,
        on_fit_start=on_fit_start,
        on_epoch_end=on_epoch_end,
        on_early_stop=on_early_stop,
    )
    manifest = build_training_artifact_manifest(training_run.prepared, spec=spec)

    if persist_artifact:
        persist_training_artifact(
            artifact_dir,
            manifest=manifest,
            model=training_run.model,
        )
        loaded_artifact = load_training_artifact(artifact_dir)
        best_validation_metrics, test_metrics = _evaluate_split_metrics(
            training_run,
            spec=spec,
            model=loaded_artifact.model,
        )
        runtime_summary = build_training_runtime_summary(
            training_run,
            prepared=training_run.prepared,
            best_validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
        )
        write_training_state(
            artifact_dir / ".spice" / "state.sqlite",
            summary=runtime_summary,
            epoch_rows=list(iter_epoch_records(training_run)),
        )
    else:
        best_validation_metrics, test_metrics = _evaluate_split_metrics(
            training_run,
            spec=spec,
            model=training_run.model,
        )
        runtime_summary = build_training_runtime_summary(
            training_run,
            prepared=training_run.prepared,
            best_validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
        )
    return PersistedTrainingRun(
        training_run=training_run,
        manifest=manifest,
        summary=LoadedTrainingSummary(manifest=manifest, runtime=runtime_summary),
        artifact_dir=artifact_dir,
    )

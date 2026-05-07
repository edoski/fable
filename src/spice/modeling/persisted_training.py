"""Persisted training orchestration and artifact/state writes."""

from __future__ import annotations

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
from .pipeline import TrainingRunCallbacks, TrainingSpec, run_training
from .results import (
    LoadedTrainingSummary,
    TrainingRuntimeSummary,
    build_training_runtime_summary,
    iter_epoch_records,
)
from .scoring import PredictionMetricScoringRuntimePlan, score_prediction_metrics
from .training_run import TrainingRunResult


@dataclass(slots=True)
class PersistedTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    summary: LoadedTrainingSummary
    artifact_dir: Path


@dataclass(slots=True)
class TrialTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    summary: LoadedTrainingSummary


def _evaluate_split_metrics(
    training_run: TrainingRunResult,
    *,
    spec: TrainingSpec,
    model,
) -> tuple[MetricSet, MetricSet]:
    prepared = training_run.prepared
    best_validation_metrics = score_prediction_metrics(
        PredictionMetricScoringRuntimePlan(
            model=model,
            prediction_contract=spec.prediction_contract,
            representation_contract=spec.representation_contract,
            execution_policy=prepared.execution_policy,
            store=prepared.store,
            temporal_facts=prepared.samples.validation.temporal_facts,
            prediction_training_state=training_run.training_result.prediction_training_state,
            runtime_plan=training_run.training_result.runtime_plan,
        )
    )
    test_metrics = score_prediction_metrics(
        PredictionMetricScoringRuntimePlan(
            model=model,
            prediction_contract=spec.prediction_contract,
            representation_contract=spec.representation_contract,
            execution_policy=prepared.execution_policy,
            store=prepared.store,
            temporal_facts=prepared.samples.test.temporal_facts,
            prediction_training_state=training_run.training_result.prediction_training_state,
            runtime_plan=training_run.training_result.runtime_plan,
        )
    )
    return best_validation_metrics, test_metrics


def _build_summary(
    training_run: TrainingRunResult,
    *,
    manifest: TrainingArtifactManifest,
    spec: TrainingSpec,
    model,
) -> LoadedTrainingSummary:
    best_validation_metrics, test_metrics = _evaluate_split_metrics(
        training_run,
        spec=spec,
        model=model,
    )
    runtime_summary = build_training_runtime_summary(
        training_run,
        prepared=training_run.prepared,
        best_validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
    )
    return LoadedTrainingSummary(manifest=manifest, runtime=runtime_summary)


def run_persisted_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    callbacks: TrainingRunCallbacks | None = None,
) -> PersistedTrainingRun:
    training_run = run_training(
        history_block_path,
        spec=spec,
        callbacks=callbacks,
    )
    manifest = build_training_artifact_manifest(training_run.prepared, spec=spec)
    persist_training_artifact(
        artifact_dir,
        manifest=manifest,
        model=training_run.model,
    )
    loaded_artifact = load_training_artifact(artifact_dir)
    summary = _build_summary(
        training_run,
        manifest=manifest,
        spec=spec,
        model=loaded_artifact.model,
    )
    _write_training_summary(
        artifact_dir,
        summary=summary.runtime,
        training_run=training_run,
    )
    return PersistedTrainingRun(
        training_run=training_run,
        manifest=manifest,
        summary=summary,
        artifact_dir=artifact_dir,
    )


def run_trial_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    callbacks: TrainingRunCallbacks | None = None,
) -> TrialTrainingRun:
    training_run = run_training(
        history_block_path,
        spec=spec,
        callbacks=callbacks,
    )
    manifest = build_training_artifact_manifest(training_run.prepared, spec=spec)
    summary = _build_summary(
        training_run,
        manifest=manifest,
        spec=spec,
        model=training_run.model,
    )
    return TrialTrainingRun(
        training_run=training_run,
        manifest=summary.manifest,
        summary=summary,
    )


def _write_training_summary(
    artifact_dir: Path,
    *,
    summary: TrainingRuntimeSummary,
    training_run: TrainingRunResult,
) -> None:
    write_training_state(
        artifact_dir / ".spice" / "state.sqlite",
        summary=summary,
        epoch_rows=list(iter_epoch_records(training_run)),
    )

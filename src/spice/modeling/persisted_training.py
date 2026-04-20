"""Persisted training orchestration and artifact/state writes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.reporting import NullReporter, Reporter
from ..prediction import MetricSet
from ..storage.artifact import write_training_state
from ..storage.engine import RootKind
from .artifacts import (
    TrainingArtifactManifest,
    build_training_artifact_manifest,
    load_training_artifact,
    persist_training_artifact,
)
from .pipeline import TrainingRunResult, TrainingSpec, TrainingStageReporters, run_training
from .results import (
    LoadedTrainingSummary,
    build_training_runtime_summary,
    iter_epoch_records,
)
from .training import evaluate_model


@dataclass(slots=True)
class PersistedTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    summary: LoadedTrainingSummary
    artifact_dir: Path
    artifact_paths: tuple[Path, ...]


def _evaluate_split_metrics(
    training_run: TrainingRunResult,
    *,
    spec: TrainingSpec,
    model,
    reporter: Reporter,
) -> tuple[MetricSet, MetricSet]:
    prepared = training_run.prepared
    best_validation_metrics = evaluate_model(
        model,
        model_config=spec.model,
        prediction_contract=spec.prediction_contract,
        realization_policy=prepared.realization_policy,
        representation_contract=spec.representation_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.validation,
        prediction_training_state=training_run.prediction_training_state,
        training_config=spec.training,
        reporter=reporter,
    )
    test_metrics = evaluate_model(
        model,
        model_config=spec.model,
        prediction_contract=spec.prediction_contract,
        realization_policy=prepared.realization_policy,
        representation_contract=spec.representation_contract,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        prediction_training_state=training_run.prediction_training_state,
        training_config=spec.training,
        reporter=reporter,
    )
    return best_validation_metrics, test_metrics


def run_persisted_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    stage_reporters: TrainingStageReporters | None = None,
    write_reporter: Reporter | None = None,
    reporter: Reporter | None = None,
    persist_artifact: bool = True,
    state_root_kind: RootKind | None = None,
) -> PersistedTrainingRun:
    reporter = reporter or NullReporter()
    active_stage_reporters = stage_reporters or TrainingStageReporters.shared(reporter)
    active_write_reporter = write_reporter or reporter
    training_run = run_training(
        history_block_path,
        spec=spec,
        artifact_dir=artifact_dir,
        stage_reporters=active_stage_reporters,
        reporter=reporter,
    )
    manifest = build_training_artifact_manifest(training_run.prepared, spec=spec)

    artifact_paths: list[Path] = []
    if persist_artifact:
        if state_root_kind is None:
            raise ValueError("state_root_kind is required when persist_artifact is true")
        artifact_task = active_write_reporter.start_task("write training artifact")
        persist_training_artifact(
            artifact_dir,
            manifest=manifest,
            root_kind=state_root_kind,
            model=training_run.model,
        )
        loaded_artifact = load_training_artifact(artifact_dir)
        best_validation_metrics, test_metrics = _evaluate_split_metrics(
            training_run,
            spec=spec,
            model=loaded_artifact.model,
            reporter=active_stage_reporters.evaluate,
        )
        runtime_summary = build_training_runtime_summary(
            training_run,
            prepared=training_run.prepared,
            best_validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
        )
        write_training_state(
            artifact_dir / ".spice" / "state.sqlite",
            root_kind=state_root_kind,
            summary=runtime_summary,
            epoch_rows=list(iter_epoch_records(training_run)),
        )
        active_write_reporter.finish_task(artifact_task, message=str(artifact_dir), silent=True)
        artifact_paths.extend(
            [
                artifact_dir / ".spice" / "state.sqlite",
                artifact_dir / "model.pt",
            ]
        )
    else:
        best_validation_metrics, test_metrics = _evaluate_split_metrics(
            training_run,
            spec=spec,
            model=training_run.model,
            reporter=active_stage_reporters.evaluate,
        )
        runtime_summary = build_training_runtime_summary(
            training_run,
            prepared=training_run.prepared,
            best_validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
        )
    if training_run.training_result.best_checkpoint_path is not None:
        artifact_paths.append(training_run.training_result.best_checkpoint_path)

    return PersistedTrainingRun(
        training_run=training_run,
        manifest=manifest,
        summary=LoadedTrainingSummary(manifest=manifest, runtime=runtime_summary),
        artifact_dir=artifact_dir,
        artifact_paths=tuple(artifact_paths),
    )

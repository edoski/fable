"""Shared training execution and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.console import NullReporter, Reporter
from ..state.artifact import write_training_state
from .artifacts import (
    TrainingArtifactManifest,
    build_training_artifact_manifest,
    persist_training_artifact,
)
from .evaluation import EpochMetrics
from .pipeline import TrainingRunResult, TrainingSpec, TrainingStageReporters, run_training
from .reporting import (
    TrainingSummary,
    build_training_summary,
    iter_epoch_pairs,
    summarize_epoch_metrics,
)


@dataclass(slots=True)
class PersistedTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    summary: TrainingSummary
    best_validation_metrics: EpochMetrics
    artifact_dir: Path
    artifact_paths: tuple[Path, ...]


def run_persisted_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    stage_reporters: TrainingStageReporters | None = None,
    write_reporter: Reporter | None = None,
    reporter: Reporter | None = None,
    persist_artifact: bool = True,
    state_root_kind: str | None = None,
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
    summary = build_training_summary(
        training_run,
        chain_name=spec.chain.name,
        dataset_id=spec.dataset_id,
        model_id=spec.model.id,
        manifest=manifest,
        prepared=training_run.prepared,
    )

    validation_history = training_run.training_result.validation_history
    if not validation_history:
        raise RuntimeError("Training did not produce validation metrics")
    best_validation_metrics = validation_history[training_run.training_result.best_epoch - 1]

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
        write_training_state(
            artifact_dir / ".spice" / "state.sqlite",
            root_kind=state_root_kind,
            summary=summary,
            epoch_rows=[
                (
                    epoch,
                    summarize_epoch_metrics(train_metrics),
                    summarize_epoch_metrics(validation_metrics),
                )
                for epoch, train_metrics, validation_metrics in iter_epoch_pairs(training_run)
            ],
        )
        active_write_reporter.finish_task(artifact_task, message=str(artifact_dir), silent=True)
        artifact_paths.extend(
            [
                artifact_dir / ".spice" / "state.sqlite",
                artifact_dir / "model.pt",
            ]
        )
    if training_run.training_result.best_checkpoint_path is not None:
        artifact_paths.append(training_run.training_result.best_checkpoint_path)

    return PersistedTrainingRun(
        training_run=training_run,
        manifest=manifest,
        summary=summary,
        best_validation_metrics=best_validation_metrics,
        artifact_dir=artifact_dir,
        artifact_paths=tuple(artifact_paths),
    )

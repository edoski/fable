"""Shared training execution and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.console import NullReporter, Reporter
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from .artifacts import (
    TrainingArtifactManifest,
    build_training_artifact_manifest,
    write_training_artifact,
)
from .evaluation import EpochMetrics
from .pipeline import TrainingRunResult, TrainingSpec, run_training
from .reporting import TrainingRunReport, build_training_run_report, write_json_report


@dataclass(slots=True)
class PersistedTrainingRun:
    training_run: TrainingRunResult
    manifest: TrainingArtifactManifest
    report: TrainingRunReport
    best_validation_metrics: EpochMetrics
    artifact_dir: Path
    report_path: Path
    artifact_paths: tuple[Path, ...]


def run_persisted_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    report_path: Path,
    reporter: Reporter | None = None,
) -> PersistedTrainingRun:
    reporter = reporter or NullReporter()
    training_run = run_training(
        history_block_path,
        spec=spec,
        artifact_dir=artifact_dir,
        reporter=reporter,
    )
    manifest = build_training_artifact_manifest(training_run.prepared, spec=spec)
    artifact_task = reporter.start_task("write training artifact")
    write_training_artifact(
        artifact_dir,
        manifest=manifest,
        model=training_run.model,
    )
    reporter.finish_task(artifact_task, message=str(artifact_dir))
    report = build_training_run_report(
        training_run,
        anchor_count=spec.anchor_count,
        max_delay_seconds=spec.max_delay_seconds,
        lookback_seconds=spec.lookback_seconds,
        chain_name=spec.chain.name.value,
        dataset_id=spec.dataset_id,
        family=spec.model.family.value,
        block_time_seconds=spec.chain.block_time_seconds,
        manifest=manifest,
        prepared=training_run.prepared,
        artifact_dir=artifact_dir,
        history_block_path=history_block_path,
        device_requested=spec.training.device,
    )
    report_task = reporter.start_task("write training report")
    write_json_report(report_path, report)
    reporter.finish_task(report_task, message=str(report_path))

    validation_history = training_run.training_result.validation_history
    if not validation_history:
        raise RuntimeError("Training did not produce validation metrics")
    best_validation_metrics = validation_history[training_run.training_result.best_epoch - 1]

    artifact_paths = [
        artifact_dir / ARTIFACT_MANIFEST_FILENAME,
        artifact_dir / MODEL_STATE_FILENAME,
        report_path,
    ]
    if training_run.training_result.best_checkpoint_path is not None:
        artifact_paths.append(training_run.training_result.best_checkpoint_path)

    return PersistedTrainingRun(
        training_run=training_run,
        manifest=manifest,
        report=report,
        best_validation_metrics=best_validation_metrics,
        artifact_dir=artifact_dir,
        report_path=report_path,
        artifact_paths=tuple(artifact_paths),
    )

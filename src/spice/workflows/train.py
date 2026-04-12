"""Training workflow."""

from __future__ import annotations

from ..config import ArtifactVariant, TrainConfig
from ..core.console import Reporter
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from ..core.files import remove_path
from ..modeling.execution import run_persisted_training
from ..modeling.pipeline import TrainingStageReporters
from ._shared import (
    abort_cleanup,
    apply_study_best_params,
    build_training_spec,
    managed_workflow,
)


def _format_train_summary_sections(
    config: TrainConfig,
    persisted,
) -> list[tuple[str, list[tuple[str, str]]]]:
    report = persisted.report
    result = persisted.training_run.training_result
    best_validation = persisted.best_validation_metrics
    return [
        (
            "dataset",
            [
                ("id", report.dataset_id),
                ("chain", report.chain),
                ("model", report.model_id),
                ("delay", f"{report.max_delay_seconds}s"),
            ],
        ),
        (
            "provenance",
            [
                ("variant", report.variant.value),
                *([] if report.study is None else [("study", report.study.id)]),
                ("artifact", str(report.artifact_dir)),
            ],
        ),
        (
            "runtime",
            [
                ("lookback", f"{report.lookback_seconds}s"),
                ("best epoch", str(report.best_epoch)),
                ("device", result.resolved_device),
                ("precision", result.resolved_precision),
                ("compile", "on" if result.compiled else "off"),
            ],
        ),
        (
            "metrics",
            [
                (
                    "split sizes",
                    (
                        f"train={report.split_sizes.train_samples:,} "
                        f"validation={report.split_sizes.validation_samples:,} "
                        f"test={report.split_sizes.test_samples:,}"
                    ),
                ),
                ("validation loss", f"{best_validation.total_loss:.4f}"),
                ("validation accuracy", f"{best_validation.accuracy:.3f}"),
                (
                    "test profit over baseline",
                    f"{report.test_metrics.mean_profit_over_baseline:.4f}",
                ),
            ],
        ),
    ]


def _clean_training_outputs(config: TrainConfig, *, prune_empty_root: bool) -> None:
    artifact_root = config.paths.artifact_root
    checkpoint_dir = config.paths.checkpoint_dir
    train_report_path = config.paths.train_report_path
    simulation_report_path = config.paths.simulation_report_path
    if (
        artifact_root is None
        or checkpoint_dir is None
        or train_report_path is None
        or simulation_report_path is None
    ):
        raise ValueError("training workflow requires artifact output paths")
    for path in (
        checkpoint_dir,
        artifact_root / ARTIFACT_MANIFEST_FILENAME,
        artifact_root / MODEL_STATE_FILENAME,
        train_report_path,
        simulation_report_path,
    ):
        remove_path(path)
    if prune_empty_root and artifact_root.exists():
        try:
            next(artifact_root.iterdir())
        except StopIteration:
            artifact_root.rmdir()


def _workflow_facts(config: TrainConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.id),
        ("chain", config.chain.name),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant is ArtifactVariant.TUNED:
        facts.append(("study", config.study.id))
    return facts


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        config,
        run_name=(
            "train-"
            f"{config.chain.name}-{config.model.id}-"
            f"{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        active_config = config
        if config.artifact.variant is ArtifactVariant.TUNED:
            active_config = apply_study_best_params(config)
        session.runtime.configure_workflow("train", _workflow_facts(active_config))
        spec = build_training_spec(active_config)
        artifact_dir = active_config.paths.artifact_root
        report_path = active_config.paths.train_report_path
        history_block_path = active_config.paths.history_dir
        if artifact_dir is None or report_path is None:
            raise ValueError("training workflow requires artifact output paths")
        stage_reporters = TrainingStageReporters(
            load=session.runtime.stage_reporter("load", label="load"),
            prepare=session.runtime.stage_reporter("prepare", label="prepare"),
            build=session.runtime.stage_reporter("build", label="build"),
            fit=session.runtime.stage_reporter(
                "fit",
                label="fit",
                running_status="running",
            ),
            evaluate=session.runtime.stage_reporter("evaluate", label="evaluate"),
        )
        write_reporter = session.runtime.stage_reporter(
            "write",
            label="write",
            running_status="writing",
        )
        with abort_cleanup(
            session.reporter,
            label="train",
            cleanup=lambda: _clean_training_outputs(active_config, prune_empty_root=True),
        ):
            _clean_training_outputs(active_config, prune_empty_root=True)
            persisted = run_persisted_training(
                history_block_path,
                spec=spec,
                artifact_dir=artifact_dir,
                report_path=report_path,
                stage_reporters=stage_reporters,
                write_reporter=write_reporter,
                reporter=session.reporter,
            )
        session.runtime.log_sectioned_summary(
            "training summary",
            _format_train_summary_sections(active_config, persisted),
        )

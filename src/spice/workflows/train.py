"""Training workflow."""

from __future__ import annotations

from ..config import ArtifactVariant, TrainConfig
from ..core.console import Reporter
from ..core.constants import MODEL_STATE_FILENAME
from ..core.files import remove_path
from ..modeling.execution import run_persisted_training
from ..modeling.pipeline import TrainingStageReporters
from ..state import ARTIFACT_ROOT_KIND
from ..state.catalog import upsert_artifact_record
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
    summary = persisted.summary
    result = persisted.training_run.training_result
    best_validation = persisted.best_validation_metrics
    return [
        (
            "dataset",
            [
                ("name", summary.dataset_name),
                ("storage id", summary.dataset_id),
                ("chain", summary.chain),
                ("model", summary.model_id),
                ("task", summary.task_id),
            ],
        ),
        (
            "provenance",
            [
                ("artifact id", summary.artifact_id),
                ("variant", summary.variant.value),
                *([] if summary.study is None else [("study", summary.study.name)]),
                ("capability", f"{summary.max_supported_delay_seconds}s"),
            ],
        ),
        (
            "runtime",
            [
                ("lookback", f"{summary.lookback_seconds}s"),
                ("best epoch", str(summary.best_epoch)),
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
                        f"train={summary.split_sizes.train_samples:,} "
                        f"validation={summary.split_sizes.validation_samples:,} "
                        f"test={summary.split_sizes.test_samples:,}"
                    ),
                ),
                ("validation loss", f"{best_validation.total_loss:.4f}"),
                ("validation accuracy", f"{best_validation.accuracy:.3f}"),
                (
                    "test profit over baseline",
                    f"{summary.test_metrics.mean_profit_over_baseline:.4f}",
                ),
            ],
        ),
    ]


def _clean_training_outputs(config: TrainConfig, *, prune_empty_root: bool) -> None:
    artifact_root = config.paths.artifact_root
    checkpoint_dir = config.paths.checkpoint_dir
    artifact_state_db = config.paths.artifact_state_db
    if artifact_root is None or checkpoint_dir is None:
        raise ValueError("training workflow requires artifact output paths")
    paths = [
        checkpoint_dir,
        artifact_root / MODEL_STATE_FILENAME,
    ]
    if artifact_state_db is not None:
        paths.append(artifact_state_db)
    for path in paths:
        remove_path(path)
    if prune_empty_root and artifact_root.exists():
        try:
            next(artifact_root.iterdir())
        except StopIteration:
            artifact_root.rmdir()


def _workflow_facts(config: TrainConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant is ArtifactVariant.TUNED:
        facts.append(("study", config.study.name))
    return facts


def _state_root_kind(config: TrainConfig) -> str:
    del config
    return ARTIFACT_ROOT_KIND


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        config,
        run_name=(
            "train-"
            f"{config.chain.name}-{config.model.id}-"
            f"{config.task.id}"
        ),
        reporter=reporter,
    ) as session:
        active_config = config
        if config.artifact.variant is ArtifactVariant.TUNED:
            active_config = apply_study_best_params(config)
        session.runtime.configure_workflow("train", _workflow_facts(active_config))
        spec = build_training_spec(active_config)
        artifact_dir = active_config.paths.artifact_root
        history_block_path = active_config.paths.history_dir
        if artifact_dir is None:
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
                stage_reporters=stage_reporters,
                write_reporter=write_reporter,
                reporter=session.reporter,
                state_root_kind=_state_root_kind(active_config),
            )
            artifact_root = active_config.paths.artifact_root
            artifact_state_db = active_config.paths.artifact_state_db
            artifact_id = active_config.paths.artifact_id
            if artifact_root is None or artifact_state_db is None or artifact_id is None:
                raise ValueError("training workflow requires artifact output paths")
            upsert_artifact_record(
                active_config.paths.catalog_db,
                artifact_id=artifact_id,
                dataset_id=active_config.paths.corpus_id,
                dataset_name=active_config.dataset.name,
                chain_name=active_config.chain.name,
                feature_set_id=active_config.feature_set.id,
                model_id=active_config.model.id,
                task_id=active_config.task.id,
                variant=active_config.artifact.variant.value,
                study_id=active_config.paths.study_id,
                study_name=(
                    active_config.study.name
                    if active_config.artifact.variant is ArtifactVariant.TUNED
                    else None
                ),
                root_path=artifact_root,
                state_db_path=artifact_state_db,
            )
        session.runtime.log_sectioned_summary(
            "training summary",
            _format_train_summary_sections(active_config, persisted),
        )

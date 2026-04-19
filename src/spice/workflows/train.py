"""Training workflow."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ..config import ArtifactVariant, TrainConfig
from ..core.errors import ConfigResolutionError
from ..core.files import promote_paths_atomic, prune_empty_directories, remove_path
from ..core.reporting import Reporter, StageMetricDescriptor
from ..modeling.persisted_training import run_persisted_training
from ..modeling.pipeline import TrainingStageReporters, build_training_spec
from ..modeling.summary import training_summary_sections
from ..modeling.tuning import apply_study_best_params
from ..storage.catalog import upsert_artifact_record
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.layout import resolve_workflow_paths
from ._shared import abort_cleanup, managed_workflow

_FIT_STAGE_METRICS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="epoch", label="epoch"),
)


def _build_staged_artifact_root(artifact_root: Path) -> Path:
    return artifact_root.parent / f".{artifact_root.name}.staging.{uuid4().hex}"


def _cleanup_staged_artifact_root(staged_root: Path, *, prune_stop_at: Path) -> None:
    remove_path(staged_root)
    prune_empty_directories(staged_root.parent, stop_at=prune_stop_at)


def _workflow_facts(config: TrainConfig) -> list[tuple[str, str]]:
    facts = [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
    ]
    if config.artifact.variant is ArtifactVariant.TUNED:
        facts.append(("study", config.study.name))
    return facts


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        reporter=reporter,
    ) as session:
        active_config = config
        if config.artifact.variant is ArtifactVariant.TUNED:
            active_config = apply_study_best_params(config)
        paths = resolve_workflow_paths(active_config)
        session.runtime.configure_workflow("train", _workflow_facts(active_config))
        spec = build_training_spec(active_config)
        artifact_dir = paths.artifact_root
        history_block_path = paths.history_dir
        if artifact_dir is None:
            raise ConfigResolutionError("training workflow requires artifact output paths")
        staged_artifact_root = _build_staged_artifact_root(artifact_dir)
        prune_stop_at = artifact_dir.parent.parent
        stage_reporters = TrainingStageReporters(
            load=session.runtime.stage_reporter("load", label="load"),
            prepare=session.runtime.stage_reporter("prepare", label="prepare"),
            build=session.runtime.stage_reporter("build", label="build"),
            fit=session.runtime.stage_reporter(
                "fit",
                label="fit",
                running_status="running",
                metric_descriptors=(
                    *_FIT_STAGE_METRICS,
                    *spec.prediction_contract.progress_metric_descriptors,
                ),
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
            cleanup=lambda: _cleanup_staged_artifact_root(
                staged_artifact_root,
                prune_stop_at=prune_stop_at,
            ),
        ):
            _cleanup_staged_artifact_root(
                staged_artifact_root,
                prune_stop_at=prune_stop_at,
            )
            persisted = run_persisted_training(
                history_block_path,
                spec=spec,
                artifact_dir=staged_artifact_root,
                stage_reporters=stage_reporters,
                write_reporter=write_reporter,
                reporter=session.reporter,
                state_root_kind=ARTIFACT_ROOT_KIND,
            )
            artifact_root = paths.artifact_root
            artifact_state_db = paths.artifact_state_db
            artifact_id = paths.artifact_id
            if artifact_root is None or artifact_state_db is None or artifact_id is None:
                raise ConfigResolutionError("training workflow requires artifact output paths")
            promote_paths_atomic([(artifact_root, staged_artifact_root)])
            upsert_artifact_record(
                paths.catalog_db,
                artifact_id=artifact_id,
                dataset_id=paths.corpus_id,
                dataset_name=active_config.dataset.name,
                chain_name=active_config.chain.name,
                feature_set_id=active_config.feature_set.id,
                prediction_id=active_config.prediction.id,
                model_id=active_config.model.id,
                problem_id=active_config.problem.id,
                variant=active_config.artifact.variant.value,
                study_id=paths.study_id,
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
            training_summary_sections(persisted.summary),
        )

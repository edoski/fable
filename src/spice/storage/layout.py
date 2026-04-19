"""Deterministic workflow storage layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, overload

from ..core.errors import ConfigResolutionError
from .ids import artifact_storage_id, corpus_storage_id, study_storage_id

if TYPE_CHECKING:
    from ..config.models import (
        AcquireConfig,
        ArtifactVariant,
        ChainSpec,
        DatasetSpec,
        ModelWorkflowConfig,
        StorageSpec,
    )

_CATALOG_DB_FILENAME = "catalog.sqlite"


@dataclass(frozen=True, slots=True)
class WorkflowPaths:
    output_root: Path
    catalog_db: Path
    corpus_id: str
    corpus_root: Path
    history_dir: Path
    evaluation_dir: Path
    corpus_state_db: Path
    artifact_id: str | None = None
    artifact_root: Path | None = None
    checkpoint_dir: Path | None = None
    artifact_state_db: Path | None = None
    study_id: str | None = None
    study_root: Path | None = None
    study_state_db: Path | None = None


def catalog_db_path(storage_root: Path) -> Path:
    return storage_root / ".spice" / _CATALOG_DB_FILENAME


def build_workflow_paths(
    *,
    storage: StorageSpec,
    chain: ChainSpec,
    dataset: DatasetSpec,
    dataset_builder_payload: dict[str, object] | None = None,
    feature_set_payload: dict[str, object] | None = None,
    model_payload: dict[str, object] | None = None,
    problem_payload: dict[str, object] | None = None,
    prediction_payload: dict[str, object] | None = None,
    variant: ArtifactVariant | None = None,
    study_name: str = "default",
    include_artifacts: bool = False,
    tuning_mode: bool = False,
) -> WorkflowPaths:
    from ..config.models import ArtifactVariant

    output_root = storage.root
    catalog_db = catalog_db_path(output_root)
    resolved_variant = ArtifactVariant.BASELINE if variant is None else variant
    corpus_id = corpus_storage_id(chain_name=chain.name, dataset_name=dataset.name)
    corpus_root = output_root / "corpora" / chain.name / corpus_id
    artifact_id: str | None = None
    artifact_root: Path | None = None
    checkpoint_dir: Path | None = None
    artifact_state_db: Path | None = None
    study_id: str | None = None
    study_root: Path | None = None
    study_state_db: Path | None = None

    if include_artifacts:
        if (
            feature_set_payload is None
            or dataset_builder_payload is None
            or model_payload is None
            or problem_payload is None
            or prediction_payload is None
        ):
            raise ConfigResolutionError(
                "artifact paths require dataset_builder_payload, "
                "feature_set_payload, model_payload, "
                "problem_payload, prediction_payload"
            )
        if tuning_mode or resolved_variant is ArtifactVariant.TUNED:
            study_id = study_storage_id(
                chain_name=chain.name,
                corpus_id=corpus_id,
                dataset_builder=dataset_builder_payload,
                feature_set=feature_set_payload,
                model=model_payload,
                problem=problem_payload,
                prediction=prediction_payload,
                study_name=study_name,
            )
            study_root = output_root / "studies" / chain.name / study_id
            study_state_db = study_root / ".spice" / "state.sqlite"
        if not tuning_mode:
            artifact_id = artifact_storage_id(
                chain_name=chain.name,
                corpus_id=corpus_id,
                dataset_builder=dataset_builder_payload,
                feature_set=feature_set_payload,
                model=model_payload,
                problem=problem_payload,
                prediction=prediction_payload,
                variant=resolved_variant.value,
                study_id=study_id if resolved_variant is ArtifactVariant.TUNED else None,
            )
            artifact_root = output_root / "artifacts" / chain.name / artifact_id
            checkpoint_dir = artifact_root / "checkpoints"
            artifact_state_db = artifact_root / ".spice" / "state.sqlite"

    return WorkflowPaths(
        output_root=output_root,
        catalog_db=catalog_db,
        corpus_id=corpus_id,
        corpus_root=corpus_root,
        history_dir=corpus_root / "history",
        evaluation_dir=corpus_root / "evaluation",
        corpus_state_db=corpus_root / ".spice" / "state.sqlite",
        artifact_id=artifact_id,
        artifact_root=artifact_root,
        checkpoint_dir=checkpoint_dir,
        artifact_state_db=artifact_state_db,
        study_id=study_id,
        study_root=study_root,
        study_state_db=study_state_db,
    )


@overload
def resolve_workflow_paths(config: AcquireConfig) -> WorkflowPaths: ...


@overload
def resolve_workflow_paths(config: ModelWorkflowConfig) -> WorkflowPaths: ...


def resolve_workflow_paths(config: object) -> WorkflowPaths:
    from ..config.models import AcquireConfig, ModelWorkflowConfig

    if isinstance(config, AcquireConfig):
        return build_workflow_paths(
            storage=config.storage,
            chain=config.chain,
            dataset=config.dataset,
        )
    if isinstance(config, ModelWorkflowConfig):
        return build_workflow_paths(
            storage=config.storage,
            chain=config.chain,
            dataset=config.dataset,
            dataset_builder_payload=config.dataset_builder.model_dump(
                mode="json",
                exclude_none=True,
            ),
            feature_set_payload=config.feature_set.model_dump(mode="json", exclude_none=True),
            model_payload=config.model.model_dump(mode="json", exclude_none=True),
            problem_payload=config.problem.model_dump(mode="json"),
            prediction_payload=config.prediction.model_dump(mode="json"),
            variant=config.artifact.variant,
            study_name=config.study.name,
            include_artifacts=True,
            tuning_mode=config.workflow.value == "tune",
        )
    raise TypeError(f"Unsupported workflow config for path resolution: {type(config)!r}")

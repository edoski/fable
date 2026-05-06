from __future__ import annotations

from pathlib import Path

from spice.storage.catalog.records import (
    CatalogArtifactRecord,
    CatalogDatasetRecord,
    CatalogStudyRecord,
)
from spice.storage.engine import state_db_path


def dataset_record(
    root_path: Path,
    *,
    dataset_id: str = "dataset-1",
    dataset_name: str = "dataset",
    chain_name: str = "ethereum",
    state_db: Path | None = None,
) -> CatalogDatasetRecord:
    return CatalogDatasetRecord(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        chain_name=chain_name,
        root_path=root_path,
        state_db_path=state_db or state_db_path(root_path),
    )


def study_record(
    root_path: Path,
    *,
    study_id: str = "study-1",
    study_name: str = "study",
    dataset_id: str = "dataset-1",
    dataset_name: str = "dataset",
    chain_name: str = "ethereum",
    features_id: str = "features",
    prediction_id: str = "prediction",
    model_id: str = "model",
    problem_id: str = "problem",
    state_db: Path | None = None,
) -> CatalogStudyRecord:
    return CatalogStudyRecord(
        study_id=study_id,
        study_name=study_name,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        chain_name=chain_name,
        features_id=features_id,
        prediction_id=prediction_id,
        model_id=model_id,
        problem_id=problem_id,
        root_path=root_path,
        state_db_path=state_db or state_db_path(root_path),
    )


def artifact_record(
    root_path: Path,
    *,
    artifact_id: str = "artifact-1",
    dataset_id: str = "dataset-1",
    dataset_name: str = "dataset",
    chain_name: str = "ethereum",
    features_id: str = "features",
    prediction_id: str = "prediction",
    model_id: str = "model",
    problem_id: str = "problem",
    variant: str = "baseline",
    study_id: str | None = None,
    study_name: str | None = None,
    state_db: Path | None = None,
) -> CatalogArtifactRecord:
    return CatalogArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        chain_name=chain_name,
        features_id=features_id,
        prediction_id=prediction_id,
        model_id=model_id,
        problem_id=problem_id,
        variant=variant,
        study_id=study_id,
        study_name=study_name,
        root_path=root_path,
        state_db_path=state_db or state_db_path(root_path),
    )

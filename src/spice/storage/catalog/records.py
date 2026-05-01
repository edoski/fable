"""Typed catalog rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class CatalogDatasetRecord:
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class CatalogStudyRecord:
    study_id: str
    study_name: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    features_id: str
    prediction_id: str
    model_id: str
    problem_id: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class CatalogArtifactRecord:
    artifact_id: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    features_id: str
    prediction_id: str
    model_id: str
    problem_id: str
    variant: str
    study_id: str | None
    study_name: str | None
    root_path: Path
    state_db_path: Path


CatalogRecord: TypeAlias = CatalogDatasetRecord | CatalogStudyRecord | CatalogArtifactRecord

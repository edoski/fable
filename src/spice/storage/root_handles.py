"""Resolved storage-root handles for workflow roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.models import ArtifactVariant
from .catalog.records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .layout import catalog_db_path


@dataclass(frozen=True, slots=True)
class StorageRootHandle:
    root_path: Path
    catalog_db_path: Path


@dataclass(frozen=True, slots=True)
class CorpusRootHandle:
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path
    history_dir: Path
    evaluation_dir: Path


@dataclass(frozen=True, slots=True)
class StudyRootHandle:
    study_id: str
    study_name: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactRootHandle:
    artifact_id: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path
    variant: ArtifactVariant
    study_id: str | None = None
    study_name: str | None = None


@dataclass(frozen=True, slots=True)
class AcquireWorkflowRoots:
    storage: StorageRootHandle
    corpus: CorpusRootHandle


@dataclass(frozen=True, slots=True)
class TuneWorkflowRoots:
    storage: StorageRootHandle
    corpus: CorpusRootHandle
    study: StudyRootHandle


@dataclass(frozen=True, slots=True)
class BaselineTrainWorkflowRoots:
    storage: StorageRootHandle
    corpus: CorpusRootHandle
    artifact: ArtifactRootHandle


@dataclass(frozen=True, slots=True)
class TunedTrainWorkflowRoots:
    storage: StorageRootHandle
    corpus: CorpusRootHandle
    study: StudyRootHandle
    artifact: ArtifactRootHandle


TrainWorkflowRoots = BaselineTrainWorkflowRoots | TunedTrainWorkflowRoots


@dataclass(frozen=True, slots=True)
class EvaluateWorkflowRoots:
    storage: StorageRootHandle
    corpus: CorpusRootHandle
    artifact: ArtifactRootHandle


def storage_root_handle(storage_root: Path) -> StorageRootHandle:
    return StorageRootHandle(
        root_path=storage_root,
        catalog_db_path=catalog_db_path(storage_root),
    )


def corpus_root_from_record(record: CatalogDatasetRecord) -> CorpusRootHandle:
    return CorpusRootHandle(
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
        history_dir=record.root_path / "history",
        evaluation_dir=record.root_path / "evaluation",
    )


def study_root_from_record(record: CatalogStudyRecord) -> StudyRootHandle:
    return StudyRootHandle(
        study_id=record.study_id,
        study_name=record.study_name,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
    )


def artifact_root_from_record(record: CatalogArtifactRecord) -> ArtifactRootHandle:
    return ArtifactRootHandle(
        artifact_id=record.artifact_id,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
        variant=ArtifactVariant(record.variant),
        study_id=record.study_id,
        study_name=record.study_name,
    )

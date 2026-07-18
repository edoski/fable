"""Storage-backed workflow root handles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.models import ArtifactVariant
from .catalog.materialization import materialize_catalog_root
from .catalog.records import CatalogArtifactRecord, CatalogCorpusRecord, CatalogStudyRecord
from .corpus import load_corpus_manifest
from .engine import state_db_path
from .layout import (
    artifact_root_path,
    corpus_blocks_dir_path,
    study_root_path,
)

if TYPE_CHECKING:
    from ..corpus.metadata import CorpusManifest


@dataclass(frozen=True, slots=True)
class CorpusRootHandle:
    storage_root: Path
    corpus_id: str
    corpus_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path
    blocks_dir: Path

    def load_manifest(self) -> CorpusManifest:
        return load_corpus_manifest(self.state_db_path)


@dataclass(frozen=True, slots=True)
class StudyRootHandle:
    storage_root: Path
    study_id: str
    study_name: str
    corpus_id: str
    corpus_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactRootHandle:
    storage_root: Path
    artifact_id: str
    corpus_id: str
    corpus_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path
    variant: ArtifactVariant
    study_id: str | None = None
    study_name: str | None = None


@dataclass(frozen=True, slots=True)
class TuneWorkflowRoots:
    corpus: CorpusRootHandle
    study: StudyRootHandle


@dataclass(frozen=True, slots=True)
class BaselineTrainWorkflowRoots:
    corpus: CorpusRootHandle
    artifact: ArtifactRootHandle


@dataclass(frozen=True, slots=True)
class TunedTrainWorkflowRoots:
    corpus: CorpusRootHandle
    study: StudyRootHandle
    artifact: ArtifactRootHandle


TrainWorkflowRoots = BaselineTrainWorkflowRoots | TunedTrainWorkflowRoots


@dataclass(frozen=True, slots=True)
class EvaluateWorkflowRoots:
    corpus: CorpusRootHandle
    artifact: ArtifactRootHandle


def corpus_root_handle_from_record(
    storage_root: Path,
    record: CatalogCorpusRecord,
) -> CorpusRootHandle:
    location = materialize_catalog_root(storage_root, record)
    return CorpusRootHandle(
        storage_root=storage_root,
        corpus_id=record.corpus_id,
        corpus_name=record.corpus_name,
        chain_name=record.chain_name,
        root_path=location.root_path,
        state_db_path=location.state_db_path,
        blocks_dir=corpus_blocks_dir_path(location.root_path),
    )


def study_root_handle_from_record(
    storage_root: Path,
    record: CatalogStudyRecord,
) -> StudyRootHandle:
    location = materialize_catalog_root(storage_root, record)
    return StudyRootHandle(
        storage_root=storage_root,
        study_id=record.study_id,
        study_name=record.study_name,
        corpus_id=record.corpus_id,
        corpus_name=record.corpus_name,
        chain_name=record.chain_name,
        root_path=location.root_path,
        state_db_path=location.state_db_path,
    )


def artifact_root_handle_from_record(
    storage_root: Path,
    record: CatalogArtifactRecord,
) -> ArtifactRootHandle:
    location = materialize_catalog_root(storage_root, record)
    return ArtifactRootHandle(
        storage_root=storage_root,
        artifact_id=record.artifact_id,
        corpus_id=record.corpus_id,
        corpus_name=record.corpus_name,
        chain_name=record.chain_name,
        root_path=location.root_path,
        state_db_path=location.state_db_path,
        variant=ArtifactVariant(record.variant),
        study_id=record.study_id,
        study_name=record.study_name,
    )


def produced_study_root_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study_id: str,
    study_name: str,
) -> StudyRootHandle:
    root_path = study_root_path(
        storage_root,
        chain_name=corpus.chain_name,
        study_id=study_id,
    )
    return StudyRootHandle(
        storage_root=storage_root,
        study_id=study_id,
        study_name=study_name,
        corpus_id=corpus.corpus_id,
        corpus_name=corpus.corpus_name,
        chain_name=corpus.chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
    )


def produced_artifact_root_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str,
    variant: ArtifactVariant,
    study: StudyRootHandle | None = None,
) -> ArtifactRootHandle:
    root_path = artifact_root_path(
        storage_root,
        chain_name=corpus.chain_name,
        artifact_id=artifact_id,
    )
    return ArtifactRootHandle(
        storage_root=storage_root,
        artifact_id=artifact_id,
        corpus_id=corpus.corpus_id,
        corpus_name=corpus.corpus_name,
        chain_name=corpus.chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
        variant=variant,
        study_id=None if study is None else study.study_id,
        study_name=None if study is None else study.study_name,
    )

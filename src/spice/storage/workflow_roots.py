"""Storage-backed workflow root handles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.models import ArtifactVariant
from .corpus import load_dataset_manifest

if TYPE_CHECKING:
    from ..corpus.metadata import DatasetManifest


@dataclass(frozen=True, slots=True)
class CorpusRootHandle:
    storage_root: Path
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path
    history_dir: Path
    evaluation_dir: Path

    def load_manifest(self) -> DatasetManifest:
        return load_dataset_manifest(self.state_db_path)


@dataclass(frozen=True, slots=True)
class StudyRootHandle:
    storage_root: Path
    study_id: str
    study_name: str
    dataset_id: str
    dataset_name: str
    chain_name: str
    root_path: Path
    state_db_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactRootHandle:
    storage_root: Path
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
    corpus: CorpusRootHandle


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

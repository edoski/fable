"""Storage-backed workflow root handles and resolution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.models import AcquireConfig, ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from .artifact import upsert_evaluation_state
from .catalog.index import (
    ReindexedCatalogRoot,
    reindex_catalog_root,
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from .catalog.records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .corpus import load_dataset_manifest
from .engine import ARTIFACT_ROOT_KIND, state_db_path
from .identity import artifact_storage_identity_from_config, study_storage_identity_from_config
from .ids import artifact_storage_id, corpus_storage_id, study_storage_id
from .layout import artifact_root_path, corpus_root_path, study_root_path
from .lifecycle import PartialRootCommit, RootStage, staged_root
from .selectors import ArtifactSelector, DatasetSelector, StudySelector

if TYPE_CHECKING:
    from ..corpus.metadata import DatasetManifest
    from ..modeling.results import EvaluationRuntimeSummary


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

    def commit_splits(
        self,
        *,
        history_source: Path | None,
        evaluation_source: Path | None,
        state_db_source: Path | None,
    ) -> ReindexedCatalogRoot:
        commit = PartialRootCommit(
            storage_root=self.storage_root,
            root_path=self.root_path,
        )
        commit.add(self.history_dir, history_source)
        commit.add(self.evaluation_dir, evaluation_source)
        commit.add(self.state_db_path, state_db_source)
        return commit.commit()


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

    def reindex(self) -> ReindexedCatalogRoot:
        return reindex_catalog_root(self.storage_root, root_path=self.root_path)


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

    @contextmanager
    def stage(self, *, purpose: str = "staging", replace: bool = True) -> Iterator[RootStage]:
        with staged_root(
            storage_root=self.storage_root,
            destination_root=self.root_path,
            expected_root_kind=ARTIFACT_ROOT_KIND,
            replace=replace,
            purpose=purpose,
            prune_stop_at=self.root_path.parent.parent,
        ) as stage:
            yield stage

    def upsert_evaluation_state(self, summary: EvaluationRuntimeSummary) -> tuple[str, int]:
        return upsert_evaluation_state(self.state_db_path, summary=summary)


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


def corpus_root_from_record(storage_root: Path, record: CatalogDatasetRecord) -> CorpusRootHandle:
    return CorpusRootHandle(
        storage_root=storage_root,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
        history_dir=record.root_path / "history",
        evaluation_dir=record.root_path / "evaluation",
    )


def study_root_from_record(storage_root: Path, record: CatalogStudyRecord) -> StudyRootHandle:
    return StudyRootHandle(
        storage_root=storage_root,
        study_id=record.study_id,
        study_name=record.study_name,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
    )


def artifact_root_from_record(
    storage_root: Path,
    record: CatalogArtifactRecord,
) -> ArtifactRootHandle:
    return ArtifactRootHandle(
        storage_root=storage_root,
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


def produced_corpus_id(config: AcquireConfig) -> str:
    return corpus_storage_id(
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        evaluation_date=config.dataset.evaluation_date,
    )


def resolve_acquire_producer_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    return AcquireWorkflowRoots(
        corpus=produced_corpus_root(
            config.storage.root,
            chain_name=config.chain.name,
            dataset_id=produced_corpus_id(config),
            dataset_name=config.dataset.name,
        ),
    )


def produced_corpus_root(
    storage_root: Path,
    *,
    chain_name: str,
    dataset_id: str,
    dataset_name: str,
) -> CorpusRootHandle:
    root_path = corpus_root_path(storage_root, chain_name=chain_name, corpus_id=dataset_id)
    return CorpusRootHandle(
        storage_root=storage_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        chain_name=chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
        history_dir=root_path / "history",
        evaluation_dir=root_path / "evaluation",
    )


def produced_study_id(config: TuneConfig) -> str:
    return study_storage_id(
        identity=study_storage_identity_from_config(config, corpus_id=config.dataset_id)
    )


def produced_study_root(
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
        dataset_id=corpus.dataset_id,
        dataset_name=corpus.dataset_name,
        chain_name=corpus.chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
    )


def produced_artifact_id(config: TrainConfig, *, dataset_id: str) -> str:
    return artifact_storage_id(
        identity=artifact_storage_identity_from_config(
            config,
            corpus_id=dataset_id,
            study_id=config.study_id,
        )
    )


def produced_artifact_root(
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
        dataset_id=corpus.dataset_id,
        dataset_name=corpus.dataset_name,
        chain_name=corpus.chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
        variant=variant,
        study_id=None if study is None else study.study_id,
        study_name=None if study is None else study.study_name,
    )


def resolve_tune_roots(config: TuneConfig) -> TuneWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    corpus = corpus_root_from_record(config.storage.root, dataset)
    study_id = produced_study_id(config)
    return TuneWorkflowRoots(
        corpus=corpus,
        study=produced_study_root(
            config.storage.root,
            corpus=corpus,
            study_id=study_id,
            study_name=config.study.name,
        ),
    )


def resolve_train_roots(
    config: TrainConfig,
) -> BaselineTrainWorkflowRoots | TunedTrainWorkflowRoots:
    if config.artifact.variant is ArtifactVariant.TUNED:
        if config.study_id is None:
            raise ValueError("tuned training requires study_id")
        study = resolve_study_record(
            config.storage.root,
            selector=StudySelector(study_id=config.study_id),
        )
        dataset = resolve_dataset_record(
            config.storage.root,
            selector=DatasetSelector(dataset_id=study.dataset_id),
        )
        corpus = corpus_root_from_record(config.storage.root, dataset)
        study_root = study_root_from_record(config.storage.root, study)
        artifact_id = produced_artifact_id(config, dataset_id=study.dataset_id)
        return TunedTrainWorkflowRoots(
            corpus=corpus,
            study=study_root,
            artifact=produced_artifact_root(
                config.storage.root,
                corpus=corpus,
                artifact_id=artifact_id,
                variant=config.artifact.variant,
                study=study_root,
            ),
        )

    if config.dataset_id is None:
        raise ValueError("baseline training requires dataset_id")
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    corpus = corpus_root_from_record(config.storage.root, dataset)
    artifact_id = produced_artifact_id(config, dataset_id=dataset.dataset_id)
    return BaselineTrainWorkflowRoots(
        corpus=corpus,
        artifact=produced_artifact_root(
            config.storage.root,
            corpus=corpus,
            artifact_id=artifact_id,
            variant=config.artifact.variant,
        ),
    )


def resolve_evaluate_roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    artifact = resolve_artifact_record(
        config.storage.root,
        selector=ArtifactSelector(artifact_id=config.artifact_id),
    )
    return EvaluateWorkflowRoots(
        corpus=corpus_root_from_record(config.storage.root, dataset),
        artifact=artifact_root_from_record(config.storage.root, artifact),
    )

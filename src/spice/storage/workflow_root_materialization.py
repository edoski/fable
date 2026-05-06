"""Storage-owned workflow root materialization."""

from __future__ import annotations

from pathlib import Path

from ..config.models import AcquireConfig, ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from .catalog.index import (
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from .catalog.records import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .engine import state_db_path
from .layout import (
    artifact_root_path,
    corpus_evaluation_dir_path,
    corpus_history_dir_path,
    corpus_root_path,
    study_root_path,
)
from .root_identity import produced_root_facts
from .selectors import ArtifactSelector, DatasetSelector, StudySelector
from .workflow_roots import (
    AcquireWorkflowRoots,
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StudyRootHandle,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
)


def materialize_acquire_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    produced = produced_root_facts(config)
    if produced.corpus_id is None:
        raise ValueError("acquire root identity did not produce corpus_id")
    return AcquireWorkflowRoots(
        corpus=_produced_corpus_root(
            config.storage.root,
            chain_name=config.chain.name,
            dataset_id=produced.corpus_id,
            dataset_name=config.dataset.name,
        ),
    )


def materialize_tune_roots(config: TuneConfig) -> TuneWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    corpus = _corpus_root_from_record(config.storage.root, dataset)
    produced = produced_root_facts(config)
    if produced.study_id is None:
        raise ValueError("tune root identity did not produce study_id")
    return TuneWorkflowRoots(
        corpus=corpus,
        study=_produced_study_root(
            config.storage.root,
            corpus=corpus,
            study_id=produced.study_id,
            study_name=config.study.name,
        ),
    )


def materialize_train_roots(config: TrainConfig) -> TrainWorkflowRoots:
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
        corpus = _corpus_root_from_record(config.storage.root, dataset)
        study_root = _study_root_from_record(config.storage.root, study)
        produced = produced_root_facts(config, dataset_id=study.dataset_id)
        if produced.artifact_id is None:
            raise ValueError("train root identity did not produce artifact_id")
        return TunedTrainWorkflowRoots(
            corpus=corpus,
            study=study_root,
            artifact=_produced_artifact_root(
                config.storage.root,
                corpus=corpus,
                artifact_id=produced.artifact_id,
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
    corpus = _corpus_root_from_record(config.storage.root, dataset)
    produced = produced_root_facts(config, dataset_id=dataset.dataset_id)
    if produced.artifact_id is None:
        raise ValueError("train root identity did not produce artifact_id")
    return BaselineTrainWorkflowRoots(
        corpus=corpus,
        artifact=_produced_artifact_root(
            config.storage.root,
            corpus=corpus,
            artifact_id=produced.artifact_id,
            variant=config.artifact.variant,
        ),
    )


def materialize_evaluate_roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    artifact = resolve_artifact_record(
        config.storage.root,
        selector=ArtifactSelector(artifact_id=config.artifact_id),
    )
    return EvaluateWorkflowRoots(
        corpus=_corpus_root_from_record(config.storage.root, dataset),
        artifact=_artifact_root_from_record(config.storage.root, artifact),
    )


def _corpus_root_from_record(
    storage_root: Path,
    record: CatalogDatasetRecord,
) -> CorpusRootHandle:
    return CorpusRootHandle(
        storage_root=storage_root,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        chain_name=record.chain_name,
        root_path=record.root_path,
        state_db_path=record.state_db_path,
        history_dir=corpus_history_dir_path(record.root_path),
        evaluation_dir=corpus_evaluation_dir_path(record.root_path),
    )


def _study_root_from_record(storage_root: Path, record: CatalogStudyRecord) -> StudyRootHandle:
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


def _artifact_root_from_record(
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


def _produced_corpus_root(
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
        history_dir=corpus_history_dir_path(root_path),
        evaluation_dir=corpus_evaluation_dir_path(root_path),
    )


def _produced_study_root(
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


def _produced_artifact_root(
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

"""Producer-root identity and handle derivation."""

from __future__ import annotations

from pathlib import Path

from ..config.models import AcquireConfig, ArtifactVariant, TrainConfig, TuneConfig
from .engine import state_db_path
from .identity import artifact_storage_identity_from_config, study_storage_identity_from_config
from .ids import artifact_storage_id, corpus_storage_id, study_storage_id
from .layout import artifact_root_path, corpus_root_path, study_root_path
from .root_handles import (
    AcquireWorkflowRoots,
    ArtifactRootHandle,
    CorpusRootHandle,
    StudyRootHandle,
    storage_root_handle,
)


def produced_corpus_id(config: AcquireConfig) -> str:
    return corpus_storage_id(
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        evaluation_date=config.dataset.evaluation_date,
    )


def resolve_acquire_producer_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    return AcquireWorkflowRoots(
        storage=storage_root_handle(config.storage.root),
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

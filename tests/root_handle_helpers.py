from __future__ import annotations

from pathlib import Path

from spice.config.models import ArtifactVariant
from spice.storage.engine import state_db_path
from spice.storage.layout import artifact_root_path, corpus_root_path, study_root_path
from spice.storage.root_handles import (
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StorageRootHandle,
    StudyRootHandle,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
)


def storage_handle(storage_root: Path) -> StorageRootHandle:
    return StorageRootHandle(
        root_path=storage_root,
        catalog_db_path=storage_root / ".spice" / "catalog.sqlite",
    )


def corpus_handle(
    storage_root: Path,
    *,
    chain_name: str = "ethereum",
    dataset_id: str = "cor_test",
    dataset_name: str = "test_dataset",
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


def study_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study_id: str = "std_test",
    study_name: str = "test_study",
) -> StudyRootHandle:
    root_path = study_root_path(storage_root, chain_name=corpus.chain_name, study_id=study_id)
    return StudyRootHandle(
        study_id=study_id,
        study_name=study_name,
        dataset_id=corpus.dataset_id,
        dataset_name=corpus.dataset_name,
        chain_name=corpus.chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
    )


def artifact_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str = "art_test",
    variant: ArtifactVariant = ArtifactVariant.BASELINE,
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


def baseline_train_roots(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str = "art_test",
) -> BaselineTrainWorkflowRoots:
    return BaselineTrainWorkflowRoots(
        storage=storage_handle(storage_root),
        corpus=corpus,
        artifact=artifact_handle(storage_root, corpus=corpus, artifact_id=artifact_id),
    )


def tuned_train_roots(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study: StudyRootHandle,
    artifact_id: str = "art_test",
) -> TunedTrainWorkflowRoots:
    return TunedTrainWorkflowRoots(
        storage=storage_handle(storage_root),
        corpus=corpus,
        study=study,
        artifact=artifact_handle(
            storage_root,
            corpus=corpus,
            artifact_id=artifact_id,
            variant=ArtifactVariant.TUNED,
            study=study,
        ),
    )


def tune_roots(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study: StudyRootHandle,
) -> TuneWorkflowRoots:
    return TuneWorkflowRoots(
        storage=storage_handle(storage_root),
        corpus=corpus,
        study=study,
    )


def evaluate_roots(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact: ArtifactRootHandle,
) -> EvaluateWorkflowRoots:
    return EvaluateWorkflowRoots(
        storage=storage_handle(storage_root),
        corpus=corpus,
        artifact=artifact,
    )

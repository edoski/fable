from __future__ import annotations

from pathlib import Path

from spice.config.models import ArtifactVariant
from spice.storage.engine import state_db_path
from spice.storage.layout import (
    artifact_root_path,
    corpus_evaluation_dir_path,
    corpus_history_dir_path,
    corpus_root_path,
    study_root_path,
)
from spice.storage.workflow_roots import (
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StudyRootHandle,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
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
        storage_root=storage_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        chain_name=chain_name,
        root_path=root_path,
        state_db_path=state_db_path(root_path),
        history_dir=corpus_history_dir_path(root_path),
        evaluation_dir=corpus_evaluation_dir_path(root_path),
    )


def study_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study_id: str = "std_test",
    study_name: str = "test_study",
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


def baseline_train_roots(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str = "art_test",
) -> BaselineTrainWorkflowRoots:
    return BaselineTrainWorkflowRoots(
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
        corpus=corpus,
        study=study,
    )


def evaluate_roots(
    *,
    corpus: CorpusRootHandle,
    artifact: ArtifactRootHandle,
) -> EvaluateWorkflowRoots:
    return EvaluateWorkflowRoots(
        corpus=corpus,
        artifact=artifact,
    )

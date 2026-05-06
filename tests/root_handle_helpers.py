from __future__ import annotations

from pathlib import Path

from spice.config.models import ArtifactVariant
from spice.storage.workflow_roots import (
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StudyRootHandle,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
    produced_artifact_root_handle,
    produced_corpus_root_handle,
    produced_study_root_handle,
)


def corpus_handle(
    storage_root: Path,
    *,
    chain_name: str = "ethereum",
    dataset_id: str = "cor_test",
    dataset_name: str = "test_dataset",
) -> CorpusRootHandle:
    return produced_corpus_root_handle(
        storage_root=storage_root,
        chain_name=chain_name,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
    )


def study_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    study_id: str = "std_test",
    study_name: str = "test_study",
) -> StudyRootHandle:
    return produced_study_root_handle(
        storage_root=storage_root,
        corpus=corpus,
        study_id=study_id,
        study_name=study_name,
    )


def artifact_handle(
    storage_root: Path,
    *,
    corpus: CorpusRootHandle,
    artifact_id: str = "art_test",
    variant: ArtifactVariant = ArtifactVariant.BASELINE,
    study: StudyRootHandle | None = None,
) -> ArtifactRootHandle:
    return produced_artifact_root_handle(
        storage_root=storage_root,
        corpus=corpus,
        artifact_id=artifact_id,
        variant=variant,
        study=study,
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

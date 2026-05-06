"""Storage-owned workflow root materialization."""

from __future__ import annotations

from ..config.models import AcquireConfig, ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from .catalog.index import (
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from .root_identity import produced_root_facts
from .selectors import ArtifactSelector, DatasetSelector, StudySelector
from .workflow_roots import (
    AcquireWorkflowRoots,
    BaselineTrainWorkflowRoots,
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
    artifact_root_handle_from_record,
    corpus_root_handle_from_record,
    produced_artifact_root_handle,
    produced_corpus_root_handle,
    produced_study_root_handle,
    study_root_handle_from_record,
)


def materialize_acquire_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    produced = produced_root_facts(config)
    if produced.corpus_id is None:
        raise ValueError("acquire root identity did not produce corpus_id")
    return AcquireWorkflowRoots(
        corpus=produced_corpus_root_handle(
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
    corpus = corpus_root_handle_from_record(config.storage.root, dataset)
    produced = produced_root_facts(config)
    if produced.study_id is None:
        raise ValueError("tune root identity did not produce study_id")
    return TuneWorkflowRoots(
        corpus=corpus,
        study=produced_study_root_handle(
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
        corpus = corpus_root_handle_from_record(config.storage.root, dataset)
        study_root = study_root_handle_from_record(config.storage.root, study)
        produced = produced_root_facts(config, dataset_id=study.dataset_id)
        if produced.artifact_id is None:
            raise ValueError("train root identity did not produce artifact_id")
        return TunedTrainWorkflowRoots(
            corpus=corpus,
            study=study_root,
            artifact=produced_artifact_root_handle(
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
    corpus = corpus_root_handle_from_record(config.storage.root, dataset)
    produced = produced_root_facts(config, dataset_id=dataset.dataset_id)
    if produced.artifact_id is None:
        raise ValueError("train root identity did not produce artifact_id")
    return BaselineTrainWorkflowRoots(
        corpus=corpus,
        artifact=produced_artifact_root_handle(
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
        corpus=corpus_root_handle_from_record(config.storage.root, dataset),
        artifact=artifact_root_handle_from_record(config.storage.root, artifact),
    )

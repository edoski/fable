"""Catalog-record root resolution for workflows that consume existing roots."""

from __future__ import annotations

from ..config.models import ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from .catalog.index import resolve_artifact_record, resolve_dataset_record, resolve_study_record
from .root_handles import (
    BaselineTrainWorkflowRoots,
    EvaluateWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
    artifact_root_from_record,
    corpus_root_from_record,
    storage_root_handle,
    study_root_from_record,
)
from .root_producer_handles import (
    produced_artifact_id,
    produced_artifact_root,
    produced_study_id,
    produced_study_root,
)
from .selectors import ArtifactSelector, DatasetSelector, StudySelector


def resolve_tune_consumer_roots(config: TuneConfig) -> TuneWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    corpus = corpus_root_from_record(dataset)
    study_id = produced_study_id(config)
    return TuneWorkflowRoots(
        storage=storage_root_handle(config.storage.root),
        corpus=corpus,
        study=produced_study_root(
            config.storage.root,
            corpus=corpus,
            study_id=study_id,
            study_name=config.study.name,
        ),
    )


def resolve_train_consumer_roots(
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
        corpus = corpus_root_from_record(dataset)
        study_root = study_root_from_record(study)
        artifact_id = produced_artifact_id(config, dataset_id=study.dataset_id)
        return TunedTrainWorkflowRoots(
            storage=storage_root_handle(config.storage.root),
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
    corpus = corpus_root_from_record(dataset)
    artifact_id = produced_artifact_id(config, dataset_id=dataset.dataset_id)
    return BaselineTrainWorkflowRoots(
        storage=storage_root_handle(config.storage.root),
        corpus=corpus,
        artifact=produced_artifact_root(
            config.storage.root,
            corpus=corpus,
            artifact_id=artifact_id,
            variant=config.artifact.variant,
        ),
    )


def resolve_evaluate_consumer_roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    artifact = resolve_artifact_record(
        config.storage.root,
        selector=ArtifactSelector(artifact_id=config.artifact_id),
    )
    return EvaluateWorkflowRoots(
        storage=storage_root_handle(config.storage.root),
        corpus=corpus_root_from_record(dataset),
        artifact=artifact_root_from_record(artifact),
    )

"""Storage-owned workflow root materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..config.models import AcquireConfig, ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from ..core.errors import ConfigResolutionError, SelectorResolutionError
from .catalog.index import (
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from .identity import artifact_storage_identity_from_config, study_storage_identity_from_config
from .ids import artifact_storage_id, corpus_storage_id, study_storage_id
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


@dataclass(frozen=True, slots=True)
class ConsumedRootFacts:
    dataset_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProducedRootFacts:
    dataset_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRootFacts:
    artifact_dataset_id: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedWorkflowRootFacts:
    consumed: ConsumedRootFacts
    produced: ProducedRootFacts
    source: SourceRootFacts = SourceRootFacts()
    consumed_study_dataset_id: str | None = None
    consumed_artifact_dataset_id: str | None = None
    produced_study_dataset_id: str | None = None
    produced_artifact_dataset_id: str | None = None


def produced_corpus_id(config: AcquireConfig) -> str:
    return corpus_storage_id(
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
        evaluation_date=config.dataset.evaluation_date,
    )


def produced_study_id(config: TuneConfig) -> str:
    return study_storage_id(
        identity=study_storage_identity_from_config(config, corpus_id=config.dataset_id)
    )


def produced_artifact_id(config: TrainConfig, *, dataset_id: str) -> str:
    return artifact_storage_id(
        identity=artifact_storage_identity_from_config(
            config,
            corpus_id=dataset_id,
            study_id=config.study_id,
        )
    )


def consumed_root_facts(
    config: TrainConfig | TuneConfig | EvaluateConfig,
) -> ConsumedRootFacts:
    if isinstance(config, TuneConfig):
        return ConsumedRootFacts(dataset_id=config.dataset_id)
    if isinstance(config, TrainConfig):
        return ConsumedRootFacts(dataset_id=config.dataset_id, study_id=config.study_id)
    return ConsumedRootFacts(dataset_id=config.dataset_id, artifact_id=config.artifact_id)


def produced_root_facts(
    config: AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig,
    *,
    dataset_id: str | None = None,
) -> ProducedRootFacts:
    if isinstance(config, AcquireConfig):
        return ProducedRootFacts(dataset_id=produced_corpus_id(config))
    if isinstance(config, TuneConfig):
        return ProducedRootFacts(study_id=produced_study_id(config))
    if isinstance(config, EvaluateConfig):
        return ProducedRootFacts()
    resolved_dataset_id = dataset_id or config.dataset_id
    if resolved_dataset_id is None:
        raise ConfigResolutionError("train produced artifact identity requires dataset_id")
    return ProducedRootFacts(
        artifact_id=produced_artifact_id(config, dataset_id=resolved_dataset_id)
    )


def materialize_workflow_root_facts(
    config: AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig,
    *,
    known_study_dataset_ids: Mapping[str, str] | None = None,
    known_artifact_dataset_ids: Mapping[str, str] | None = None,
    artifact_source_dataset_id: str | None = None,
) -> MaterializedWorkflowRootFacts:
    if isinstance(config, AcquireConfig):
        return MaterializedWorkflowRootFacts(
            consumed=ConsumedRootFacts(),
            produced=produced_root_facts(config),
        )

    consumed = consumed_root_facts(config)
    study_dataset_id = (
        _study_dataset_id(
            config,
            consumed.study_id,
            known_study_dataset_ids=known_study_dataset_ids,
        )
        if consumed.study_id is not None
        else None
    )
    artifact_dataset_id = (
        _artifact_dataset_id(
            config,
            consumed.artifact_id,
            known_artifact_dataset_ids=known_artifact_dataset_ids,
            artifact_source_dataset_id=artifact_source_dataset_id,
        )
        if consumed.artifact_id is not None
        else None
    )
    train_dataset_id = _train_dataset_id(config, study_dataset_id=study_dataset_id)
    produced = produced_root_facts(config, dataset_id=train_dataset_id)
    produced_study_dataset_id = consumed.dataset_id if produced.study_id is not None else None
    produced_artifact_dataset_id = (
        train_dataset_id if produced.artifact_id is not None else None
    )
    return MaterializedWorkflowRootFacts(
        consumed=consumed,
        produced=produced,
        source=SourceRootFacts(artifact_dataset_id=artifact_source_dataset_id),
        consumed_study_dataset_id=study_dataset_id,
        consumed_artifact_dataset_id=artifact_dataset_id,
        produced_study_dataset_id=produced_study_dataset_id,
        produced_artifact_dataset_id=produced_artifact_dataset_id,
    )


def _train_dataset_id(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    *,
    study_dataset_id: str | None,
) -> str | None:
    if not isinstance(config, TrainConfig):
        return None
    if config.dataset_id is not None:
        return config.dataset_id
    if study_dataset_id is not None:
        return study_dataset_id
    raise ConfigResolutionError("train artifact source did not declare dataset_id or study_id")


def _study_dataset_id(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    study_id: str,
    *,
    known_study_dataset_ids: Mapping[str, str] | None,
) -> str:
    if known_study_dataset_ids is not None and study_id in known_study_dataset_ids:
        return known_study_dataset_ids[study_id]
    try:
        study = resolve_study_record(
            config.storage.root,
            selector=StudySelector(study_id=study_id),
        )
    except SelectorResolutionError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return study.dataset_id


def _artifact_dataset_id(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    artifact_id: str,
    *,
    known_artifact_dataset_ids: Mapping[str, str] | None,
    artifact_source_dataset_id: str | None,
) -> str:
    if artifact_source_dataset_id is not None:
        return artifact_source_dataset_id
    if known_artifact_dataset_ids is not None and artifact_id in known_artifact_dataset_ids:
        return known_artifact_dataset_ids[artifact_id]
    try:
        artifact = resolve_artifact_record(
            config.storage.root,
            selector=ArtifactSelector(artifact_id=artifact_id),
        )
    except SelectorResolutionError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return artifact.dataset_id


def materialize_acquire_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    produced = produced_root_facts(config)
    if produced.dataset_id is None:
        raise ValueError("acquire root identity did not produce dataset_id")
    return AcquireWorkflowRoots(
        corpus=produced_corpus_root_handle(
            config.storage.root,
            chain_name=config.chain.name,
            dataset_id=produced.dataset_id,
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

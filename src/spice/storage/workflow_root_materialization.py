"""Storage-owned workflow root materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..config.models import ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from ..core.errors import ConfigResolutionError, SelectorResolutionError
from .catalog.index import (
    resolve_artifact_record,
    resolve_corpus_record,
    resolve_study_record,
)
from .identity import artifact_storage_identity_from_config, study_storage_identity_from_config
from .ids import artifact_storage_id, study_storage_id
from .selectors import ArtifactSelector, CorpusSelector, StudySelector
from .workflow_roots import (
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StudyRootHandle,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
    artifact_root_handle_from_record,
    corpus_root_handle_from_record,
    produced_artifact_root_handle,
    produced_study_root_handle,
    study_root_handle_from_record,
)


@dataclass(frozen=True, slots=True)
class ConsumedRootFacts:
    corpus_id: str | None = None
    study_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProducedRootFacts:
    study_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRootFacts:
    artifact_corpus_id: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedWorkflowRootFacts:
    consumed: ConsumedRootFacts
    produced: ProducedRootFacts
    source: SourceRootFacts = SourceRootFacts()
    consumed_study_corpus_id: str | None = None
    consumed_artifact_corpus_id: str | None = None
    produced_study_corpus_id: str | None = None
    produced_artifact_corpus_id: str | None = None


@dataclass(frozen=True, slots=True)
class _WorkflowRootMaterialization:
    facts: MaterializedWorkflowRootFacts
    corpus: CorpusRootHandle | None = None
    study: StudyRootHandle | None = None
    artifact: ArtifactRootHandle | None = None


def produced_study_id(config: TuneConfig) -> str:
    return study_storage_id(
        identity=study_storage_identity_from_config(config, corpus_id=config.corpus_id)
    )


def produced_artifact_id(config: TrainConfig, *, corpus_id: str) -> str:
    return artifact_storage_id(
        identity=artifact_storage_identity_from_config(
            config,
            corpus_id=corpus_id,
            study_id=config.study_id,
        )
    )


def materialize_workflow_root_facts(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    *,
    known_study_corpus_ids: Mapping[str, str] | None = None,
    known_artifact_corpus_ids: Mapping[str, str] | None = None,
    artifact_source_corpus_id: str | None = None,
) -> MaterializedWorkflowRootFacts:
    return _materialize_workflow_roots(
        config,
        include_handles=False,
        known_study_corpus_ids=known_study_corpus_ids,
        known_artifact_corpus_ids=known_artifact_corpus_ids,
        artifact_source_corpus_id=artifact_source_corpus_id,
    ).facts


def _materialize_workflow_roots(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    *,
    include_handles: bool,
    known_study_corpus_ids: Mapping[str, str] | None = None,
    known_artifact_corpus_ids: Mapping[str, str] | None = None,
    artifact_source_corpus_id: str | None = None,
) -> _WorkflowRootMaterialization:
    if isinstance(config, TuneConfig):
        produced = ProducedRootFacts(study_id=produced_study_id(config))
        corpus = None
        study = None
        if include_handles:
            corpus = resolve_corpus_record(
                config.storage.root,
                selector=CorpusSelector(corpus_id=config.corpus_id),
            )
            corpus = corpus_root_handle_from_record(config.storage.root, corpus)
            study = produced_study_root_handle(
                config.storage.root,
                corpus=corpus,
                study_id=_required(
                    produced.study_id,
                    "tune root identity did not produce study_id",
                ),
                study_name=config.study.name,
            )
        return _WorkflowRootMaterialization(
            facts=MaterializedWorkflowRootFacts(
                consumed=ConsumedRootFacts(corpus_id=config.corpus_id),
                produced=produced,
                produced_study_corpus_id=config.corpus_id,
            ),
            corpus=corpus,
            study=study,
        )

    if isinstance(config, TrainConfig):
        if config.artifact.variant is ArtifactVariant.TUNED:
            return _materialize_tuned_train_roots(
                config,
                include_handles=include_handles,
                known_study_corpus_ids=known_study_corpus_ids,
            )
        return _materialize_baseline_train_roots(config, include_handles=include_handles)

    return _materialize_evaluate_roots(
        config,
        include_handles=include_handles,
        known_artifact_corpus_ids=known_artifact_corpus_ids,
        artifact_source_corpus_id=artifact_source_corpus_id,
    )


def _materialize_baseline_train_roots(
    config: TrainConfig,
    *,
    include_handles: bool,
) -> _WorkflowRootMaterialization:
    if config.corpus_id is None:
        raise ValueError("baseline training requires corpus_id")
    produced = ProducedRootFacts(
        artifact_id=produced_artifact_id(config, corpus_id=config.corpus_id)
    )
    corpus = None
    artifact = None
    if include_handles:
        corpus = resolve_corpus_record(
            config.storage.root,
            selector=CorpusSelector(corpus_id=config.corpus_id),
        )
        corpus = corpus_root_handle_from_record(config.storage.root, corpus)
        artifact = produced_artifact_root_handle(
            config.storage.root,
            corpus=corpus,
            artifact_id=_required(
                produced.artifact_id,
                "train root identity did not produce artifact_id",
            ),
            variant=config.artifact.variant,
        )
    return _WorkflowRootMaterialization(
        facts=MaterializedWorkflowRootFacts(
            consumed=ConsumedRootFacts(corpus_id=config.corpus_id),
            produced=produced,
            produced_artifact_corpus_id=config.corpus_id,
        ),
        corpus=corpus,
        artifact=artifact,
    )


def _materialize_tuned_train_roots(
    config: TrainConfig,
    *,
    include_handles: bool,
    known_study_corpus_ids: Mapping[str, str] | None,
) -> _WorkflowRootMaterialization:
    if config.study_id is None:
        raise ValueError("tuned training requires study_id")
    study_record = (
        resolve_study_record(
            config.storage.root,
            selector=StudySelector(study_id=config.study_id),
        )
        if include_handles
        else None
    )
    study_corpus_id = (
        study_record.corpus_id
        if study_record is not None
        else _known_or_catalog_study_corpus_id(
            config,
            config.study_id,
            known_study_corpus_ids=known_study_corpus_ids,
        )
    )
    produced = ProducedRootFacts(
        artifact_id=produced_artifact_id(config, corpus_id=study_corpus_id)
    )
    corpus = None
    study = None
    artifact = None
    if include_handles:
        if study_record is None:
            raise ValueError("tuned train study record was not resolved")
        corpus = resolve_corpus_record(
            config.storage.root,
            selector=CorpusSelector(corpus_id=study_corpus_id),
        )
        corpus = corpus_root_handle_from_record(config.storage.root, corpus)
        study = study_root_handle_from_record(config.storage.root, study_record)
        artifact = produced_artifact_root_handle(
            config.storage.root,
            corpus=corpus,
            artifact_id=_required(
                produced.artifact_id,
                "train root identity did not produce artifact_id",
            ),
            variant=config.artifact.variant,
            study=study,
        )
    return _WorkflowRootMaterialization(
        facts=MaterializedWorkflowRootFacts(
            consumed=ConsumedRootFacts(study_id=config.study_id),
            produced=produced,
            consumed_study_corpus_id=study_corpus_id,
            produced_artifact_corpus_id=study_corpus_id,
        ),
        corpus=corpus,
        study=study,
        artifact=artifact,
    )


def _materialize_evaluate_roots(
    config: EvaluateConfig,
    *,
    include_handles: bool,
    known_artifact_corpus_ids: Mapping[str, str] | None,
    artifact_source_corpus_id: str | None,
) -> _WorkflowRootMaterialization:
    artifact_record = (
        resolve_artifact_record(
            config.storage.root,
            selector=ArtifactSelector(artifact_id=config.artifact_id),
        )
        if include_handles
        else None
    )
    artifact_corpus_id = (
        artifact_record.corpus_id
        if artifact_record is not None
        else _known_or_catalog_artifact_corpus_id(
            config,
            config.artifact_id,
            known_artifact_corpus_ids=known_artifact_corpus_ids,
            artifact_source_corpus_id=artifact_source_corpus_id,
        )
    )
    corpus = None
    artifact = None
    if include_handles:
        corpus = resolve_corpus_record(
            config.storage.root,
            selector=CorpusSelector(corpus_id=config.corpus_id),
        )
        if artifact_record is None:
            raise ValueError("evaluate artifact record was not resolved")
        corpus = corpus_root_handle_from_record(config.storage.root, corpus)
        artifact = artifact_root_handle_from_record(config.storage.root, artifact_record)
    return _WorkflowRootMaterialization(
        facts=MaterializedWorkflowRootFacts(
            consumed=ConsumedRootFacts(
                corpus_id=config.corpus_id,
                artifact_id=config.artifact_id,
            ),
            produced=ProducedRootFacts(),
            source=SourceRootFacts(artifact_corpus_id=artifact_source_corpus_id),
            consumed_artifact_corpus_id=artifact_corpus_id,
        ),
        corpus=corpus,
        artifact=artifact,
    )


def _known_or_catalog_study_corpus_id(
    config: TrainConfig,
    study_id: str,
    *,
    known_study_corpus_ids: Mapping[str, str] | None,
) -> str:
    if known_study_corpus_ids is not None and study_id in known_study_corpus_ids:
        return known_study_corpus_ids[study_id]
    try:
        study = resolve_study_record(
            config.storage.root,
            selector=StudySelector(study_id=study_id),
        )
    except SelectorResolutionError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return study.corpus_id


def _known_or_catalog_artifact_corpus_id(
    config: EvaluateConfig,
    artifact_id: str,
    *,
    known_artifact_corpus_ids: Mapping[str, str] | None,
    artifact_source_corpus_id: str | None,
) -> str:
    if artifact_source_corpus_id is not None:
        return artifact_source_corpus_id
    if known_artifact_corpus_ids is not None and artifact_id in known_artifact_corpus_ids:
        return known_artifact_corpus_ids[artifact_id]
    try:
        artifact = resolve_artifact_record(
            config.storage.root,
            selector=ArtifactSelector(artifact_id=artifact_id),
        )
    except SelectorResolutionError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return artifact.corpus_id


def _required(value: str | None, message: str) -> str:
    if value is None:
        raise ValueError(message)
    return value


def materialize_tune_roots(config: TuneConfig) -> TuneWorkflowRoots:
    materialized = _materialize_workflow_roots(config, include_handles=True)
    corpus = materialized.corpus
    study = materialized.study
    if corpus is None or study is None:
        raise ValueError("tune root identity did not produce workflow roots")
    return TuneWorkflowRoots(
        corpus=corpus,
        study=study,
    )


def materialize_train_roots(config: TrainConfig) -> TrainWorkflowRoots:
    materialized = _materialize_workflow_roots(config, include_handles=True)
    corpus = materialized.corpus
    artifact = materialized.artifact
    if corpus is None or artifact is None:
        raise ValueError("train root identity did not produce workflow roots")
    if config.artifact.variant is ArtifactVariant.TUNED:
        study = materialized.study
        if study is None:
            raise ValueError("tuned train root identity did not produce study root")
        return TunedTrainWorkflowRoots(
            corpus=corpus,
            study=study,
            artifact=artifact,
        )
    return BaselineTrainWorkflowRoots(
        corpus=corpus,
        artifact=artifact,
    )


def materialize_evaluate_roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    materialized = _materialize_workflow_roots(config, include_handles=True)
    corpus = materialized.corpus
    artifact = materialized.artifact
    if corpus is None or artifact is None:
        raise ValueError("evaluate root identity did not produce workflow roots")
    return EvaluateWorkflowRoots(
        corpus=corpus,
        artifact=artifact,
    )

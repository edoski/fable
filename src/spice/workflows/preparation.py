"""Prepared workflow inputs and root resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..config.models import AcquireConfig, ArtifactVariant, EvaluateConfig, TrainConfig, TuneConfig
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..corpus.metadata import DatasetManifest
from ..modeling.artifact_inference import (
    ArtifactInferenceContext,
    prepare_artifact_inference_context,
)
from ..modeling.pipeline import TrainingSpec, build_artifact_training_spec
from ..modeling.tuning import apply_study_best_params
from ..modeling.tuning_execution import build_tuning_coverage_spec
from ..storage.catalog.index import (
    resolve_artifact_record,
    resolve_dataset_record,
    resolve_study_record,
)
from ..storage.engine import state_db_path
from ..storage.identity import (
    artifact_storage_identity_from_config,
    study_storage_identity_from_config,
)
from ..storage.ids import artifact_storage_id, corpus_storage_id, study_storage_id
from ..storage.layout import artifact_root_path, corpus_root_path, study_root_path
from ..storage.selectors import ArtifactSelector, DatasetSelector, StudySelector
from ..storage.workflow_roots import (
    AcquireWorkflowRoots,
    ArtifactRootHandle,
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    EvaluateWorkflowRoots,
    StudyRootHandle,
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    TuneWorkflowRoots,
    artifact_root_from_record,
    corpus_root_from_record,
    study_root_from_record,
)


@dataclass(frozen=True, slots=True)
class PreparedAcquireWorkflow:
    config: AcquireConfig
    roots: AcquireWorkflowRoots


@dataclass(frozen=True, slots=True)
class PreparedTrainWorkflow:
    requested_config: TrainConfig
    active_config: TrainConfig
    roots: TrainWorkflowRoots
    corpus_manifest: DatasetManifest
    spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class PreparedTuneWorkflow:
    config: TuneConfig
    roots: TuneWorkflowRoots
    corpus_manifest: DatasetManifest
    coverage_spec: TrainingSpec


@dataclass(frozen=True, slots=True)
class PreparedEvaluateWorkflow:
    config: EvaluateConfig
    roots: EvaluateWorkflowRoots
    inference_context: ArtifactInferenceContext


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


def resolve_acquire_producer_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    return AcquireWorkflowRoots(
        corpus=produced_corpus_root(
            config.storage.root,
            chain_name=config.chain.name,
            dataset_id=produced_corpus_id(config),
            dataset_name=config.dataset.name,
        ),
    )


def resolve_tune_roots(config: TuneConfig) -> TuneWorkflowRoots:
    dataset = resolve_dataset_record(
        config.storage.root,
        selector=DatasetSelector(dataset_id=config.dataset_id),
    )
    corpus = corpus_root_from_record(config.storage.root, dataset)
    return TuneWorkflowRoots(
        corpus=corpus,
        study=produced_study_root(
            config.storage.root,
            corpus=corpus,
            study_id=produced_study_id(config),
            study_name=config.study.name,
        ),
    )


def resolve_train_roots(config: TrainConfig) -> TrainWorkflowRoots:
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


def prepare_acquire(config: AcquireConfig) -> PreparedAcquireWorkflow:
    return PreparedAcquireWorkflow(
        config=config,
        roots=resolve_acquire_producer_roots(config),
    )


def prepare_train(config: TrainConfig) -> PreparedTrainWorkflow:
    roots = resolve_train_roots(config)
    active_config = config
    if config.artifact.variant is ArtifactVariant.TUNED:
        assert isinstance(roots, TunedTrainWorkflowRoots)
        applied = apply_study_best_params(
            config,
            study=roots.study,
            corpus=roots.corpus,
        )
        active_config = cast(TrainConfig, applied.config)
    corpus_manifest = roots.corpus.load_manifest()
    spec = build_artifact_training_spec(
        active_config,
        corpus=roots.corpus,
        artifact=roots.artifact,
        corpus_manifest=corpus_manifest,
    )
    validate_corpus_coverage(
        corpus_manifest,
        contract=spec.problem_contract,
        feature_contract=spec.feature_contract,
        requirement=training_coverage_requirement(spec.problem_contract),
    )
    return PreparedTrainWorkflow(
        requested_config=config,
        active_config=active_config,
        roots=roots,
        corpus_manifest=corpus_manifest,
        spec=spec,
    )


def prepare_tune(config: TuneConfig) -> PreparedTuneWorkflow:
    roots = resolve_tune_roots(config)
    corpus_manifest = roots.corpus.load_manifest()
    coverage_spec = build_tuning_coverage_spec(
        config,
        roots=roots,
        corpus_manifest=corpus_manifest,
    )
    validate_corpus_coverage(
        corpus_manifest,
        contract=coverage_spec.problem_contract,
        feature_contract=coverage_spec.feature_contract,
        requirement=training_coverage_requirement(coverage_spec.problem_contract),
    )
    return PreparedTuneWorkflow(
        config=config,
        roots=roots,
        corpus_manifest=corpus_manifest,
        coverage_spec=coverage_spec,
    )


def prepare_evaluate(config: EvaluateConfig) -> PreparedEvaluateWorkflow:
    roots = resolve_evaluate_roots(config)
    return PreparedEvaluateWorkflow(
        config=config,
        roots=roots,
        inference_context=prepare_artifact_inference_context(
            config,
            corpus=roots.corpus,
            artifact=roots.artifact,
        ),
    )

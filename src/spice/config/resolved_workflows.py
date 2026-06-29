"""Typed resolved workflow field policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast, overload

from ..evaluation import EvaluatorConfig
from ..modeling.families.base import ModelConfig
from .models import (
    ArtifactConfig,
    ChainSpec,
    CorpusSpec,
    EvaluateConfig,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
    RuntimeEvaluationWindowSpec,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
    WorkflowTask,
)

ResolvedWorkflowConfig: TypeAlias = TrainConfig | TuneConfig | EvaluateConfig
SUPPORTED_RESOLVED_WORKFLOWS = (
    WorkflowTask.TRAIN,
    WorkflowTask.TUNE,
    WorkflowTask.EVALUATE,
)
DEFAULT_EVALUATE_BATCH_SIZE = 256


@dataclass(frozen=True, slots=True)
class ResolvedModelWorkflowFields:
    chain: ChainSpec
    corpus: CorpusSpec
    storage: StorageSpec
    problem: ProblemSpec
    model: ModelConfig[str]
    features: FeaturesConfig
    prediction: PredictionConfig
    study: StudyConfig
    artifact: ArtifactConfig
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig | None
    tuning_space: TuningSpaceConfig | None


@dataclass(frozen=True, slots=True)
class ResolvedTrainWorkflowFields:
    model_fields: ResolvedModelWorkflowFields
    corpus_id: str | None
    study_id: str | None
    training_cutoff_timestamp: int | None


@dataclass(frozen=True, slots=True)
class ResolvedTuneWorkflowFields:
    model_fields: ResolvedModelWorkflowFields
    corpus_id: str
    training_cutoff_timestamp: int | None


@dataclass(frozen=True, slots=True)
class ResolvedEvaluateWorkflowFields:
    storage: StorageSpec
    artifact_id: str
    corpus_id: str
    evaluation_window: RuntimeEvaluationWindowSpec
    evaluator: EvaluatorConfig
    delay_seconds: int | None
    batch_size: int


ResolvedWorkflowFields: TypeAlias = (
    ResolvedTrainWorkflowFields
    | ResolvedTuneWorkflowFields
    | ResolvedEvaluateWorkflowFields
)
_MODEL_WORKFLOW_FIELD_NAMES = frozenset(ResolvedModelWorkflowFields.__dataclass_fields__)
_TRAIN_FIELD_NAMES = _MODEL_WORKFLOW_FIELD_NAMES | {
    "workflow",
    "corpus_id",
    "study_id",
    "training_cutoff_timestamp",
}
_TUNE_FIELD_NAMES = _MODEL_WORKFLOW_FIELD_NAMES | {
    "workflow",
    "corpus_id",
    "training_cutoff_timestamp",
}
_EVALUATE_FIELD_NAMES = frozenset(ResolvedEvaluateWorkflowFields.__dataclass_fields__) | {
    "workflow"
}


def final_evaluate_batch_size(value: int | None) -> int:
    return DEFAULT_EVALUATE_BATCH_SIZE if value is None else value


@overload
def assemble_resolved_workflow_config(fields: ResolvedTrainWorkflowFields) -> TrainConfig: ...


@overload
def assemble_resolved_workflow_config(fields: ResolvedTuneWorkflowFields) -> TuneConfig: ...


@overload
def assemble_resolved_workflow_config(
    fields: ResolvedEvaluateWorkflowFields,
) -> EvaluateConfig: ...


def assemble_resolved_workflow_config(
    fields: ResolvedWorkflowFields,
) -> ResolvedWorkflowConfig:
    if isinstance(fields, ResolvedTrainWorkflowFields):
        return _assemble_train_config(fields)
    if isinstance(fields, ResolvedTuneWorkflowFields):
        return _assemble_tune_config(fields)
    return _assemble_evaluate_config(fields)


def _assemble_train_config(fields: ResolvedTrainWorkflowFields) -> TrainConfig:
    model_fields = fields.model_fields
    return TrainConfig(
        chain=model_fields.chain,
        corpus=model_fields.corpus,
        storage=model_fields.storage,
        corpus_id=fields.corpus_id,
        study_id=fields.study_id,
        training_cutoff_timestamp=fields.training_cutoff_timestamp,
        problem=model_fields.problem,
        model=model_fields.model,
        features=model_fields.features,
        prediction=model_fields.prediction,
        study=model_fields.study,
        artifact=model_fields.artifact,
        training=model_fields.training,
        split=model_fields.split,
        tuning=model_fields.tuning,
        tuning_space=model_fields.tuning_space,
    )


def _assemble_tune_config(fields: ResolvedTuneWorkflowFields) -> TuneConfig:
    model_fields = fields.model_fields
    if model_fields.tuning is None or model_fields.tuning_space is None:
        raise ValueError("tune workflow requires tuning and tuning_space")
    return TuneConfig(
        chain=model_fields.chain,
        corpus=model_fields.corpus,
        storage=model_fields.storage,
        corpus_id=fields.corpus_id,
        training_cutoff_timestamp=fields.training_cutoff_timestamp,
        problem=model_fields.problem,
        model=model_fields.model,
        features=model_fields.features,
        prediction=model_fields.prediction,
        study=model_fields.study,
        artifact=model_fields.artifact,
        training=model_fields.training,
        split=model_fields.split,
        tuning=model_fields.tuning,
        tuning_space=model_fields.tuning_space,
    )


def _assemble_evaluate_config(fields: ResolvedEvaluateWorkflowFields) -> EvaluateConfig:
    return EvaluateConfig(
        storage=fields.storage,
        artifact_id=fields.artifact_id,
        corpus_id=fields.corpus_id,
        evaluation_window=fields.evaluation_window,
        evaluator=fields.evaluator,
        delay_seconds=fields.delay_seconds,
        batch_size=fields.batch_size,
    )


def resolved_workflow_field_names(workflow: WorkflowTask) -> frozenset[str]:
    if workflow is WorkflowTask.TRAIN:
        return _TRAIN_FIELD_NAMES
    if workflow is WorkflowTask.TUNE:
        return _TUNE_FIELD_NAMES
    if workflow is WorkflowTask.EVALUATE:
        return _EVALUATE_FIELD_NAMES
    raise ValueError(f"Unsupported resolved workflow: {workflow.value}")


def resolved_workflow_snapshot_payload(config: ResolvedWorkflowConfig) -> dict[str, object]:
    payload = config.model_dump(mode="json", exclude_none=True)
    fields = resolved_workflow_field_names(config.workflow)
    return cast(dict[str, object], {key: value for key, value in payload.items() if key in fields})

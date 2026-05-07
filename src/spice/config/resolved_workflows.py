"""Typed resolved workflow field policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from ..evaluation import EvaluatorConfig
from ..modeling.dataset_builders import DatasetBuilderConfig
from ..modeling.families.base import ModelConfig
from ..objectives import ObjectiveConfig
from .models import (
    ArtifactConfig,
    ChainSpec,
    DatasetSpec,
    EvaluateConfig,
    FeaturesConfig,
    PredictionConfig,
    ProblemSpec,
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


@dataclass(frozen=True, slots=True)
class ResolvedModelWorkflowFields:
    chain: ChainSpec
    dataset: DatasetSpec
    storage: StorageSpec
    problem: ProblemSpec
    model: ModelConfig[str]
    dataset_builder: DatasetBuilderConfig
    features: FeaturesConfig
    prediction: PredictionConfig
    objective: ObjectiveConfig
    evaluation: EvaluatorConfig | None
    study: StudyConfig
    artifact: ArtifactConfig
    split: SplitConfig
    training: TrainingConfig
    tuning: TuningConfig | None
    tuning_space: TuningSpaceConfig | None


@dataclass(frozen=True, slots=True)
class ResolvedTrainWorkflowFields:
    model_fields: ResolvedModelWorkflowFields
    dataset_id: str | None
    study_id: str | None


@dataclass(frozen=True, slots=True)
class ResolvedTuneWorkflowFields:
    model_fields: ResolvedModelWorkflowFields
    dataset_id: str


@dataclass(frozen=True, slots=True)
class ResolvedEvaluateWorkflowFields:
    storage: StorageSpec
    artifact_id: str
    dataset_id: str
    evaluation: EvaluatorConfig
    delay_seconds: int | None
    batch_size: int


_MODEL_WORKFLOW_FIELD_NAMES = frozenset(ResolvedModelWorkflowFields.__dataclass_fields__)
_TRAIN_FIELD_NAMES = _MODEL_WORKFLOW_FIELD_NAMES | {"workflow", "dataset_id", "study_id"}
_TUNE_FIELD_NAMES = _MODEL_WORKFLOW_FIELD_NAMES | {"workflow", "dataset_id"}
_EVALUATE_FIELD_NAMES = frozenset(ResolvedEvaluateWorkflowFields.__dataclass_fields__) | {
    "workflow"
}


def build_train_config(fields: ResolvedTrainWorkflowFields) -> TrainConfig:
    model_fields = fields.model_fields
    return TrainConfig(
        chain=model_fields.chain,
        dataset=model_fields.dataset,
        storage=model_fields.storage,
        dataset_id=fields.dataset_id,
        study_id=fields.study_id,
        problem=model_fields.problem,
        model=model_fields.model,
        dataset_builder=model_fields.dataset_builder,
        features=model_fields.features,
        prediction=model_fields.prediction,
        objective=model_fields.objective,
        evaluation=model_fields.evaluation,
        study=model_fields.study,
        artifact=model_fields.artifact,
        training=model_fields.training,
        split=model_fields.split,
        tuning=model_fields.tuning,
        tuning_space=model_fields.tuning_space,
    )


def build_tune_config(fields: ResolvedTuneWorkflowFields) -> TuneConfig:
    model_fields = fields.model_fields
    if model_fields.tuning is None or model_fields.tuning_space is None:
        raise ValueError("tune workflow requires tuning and tuning_space")
    return TuneConfig(
        chain=model_fields.chain,
        dataset=model_fields.dataset,
        storage=model_fields.storage,
        dataset_id=fields.dataset_id,
        problem=model_fields.problem,
        model=model_fields.model,
        dataset_builder=model_fields.dataset_builder,
        features=model_fields.features,
        prediction=model_fields.prediction,
        objective=model_fields.objective,
        evaluation=model_fields.evaluation,
        study=model_fields.study,
        artifact=model_fields.artifact,
        training=model_fields.training,
        split=model_fields.split,
        tuning=model_fields.tuning,
        tuning_space=model_fields.tuning_space,
    )


def build_evaluate_config(fields: ResolvedEvaluateWorkflowFields) -> EvaluateConfig:
    return EvaluateConfig(
        storage=fields.storage,
        artifact_id=fields.artifact_id,
        dataset_id=fields.dataset_id,
        evaluation=fields.evaluation,
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

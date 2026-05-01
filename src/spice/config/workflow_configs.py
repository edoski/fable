"""Resolved Workflow Config coercion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from pydantic import ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import coerce_objective_config
from .models import (
    AcquireConfig,
    AcquisitionConfig,
    ArtifactConfig,
    ChainSpec,
    ConfigModel,
    DatasetSpec,
    EvaluateConfig,
    PredictionConfig,
    ResolvedRpcEndpointConfig,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    WorkflowTask,
    coerce_features_config,
    coerce_problem_spec,
)

WorkflowConfig: TypeAlias = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig
ResolvedWorkflowConfig: TypeAlias = TrainConfig | TuneConfig | EvaluateConfig

ConfigModelT = TypeVar("ConfigModelT", bound=ConfigModel)


def coerce_resolved_workflow_config(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> WorkflowConfig:
    try:
        if workflow is WorkflowTask.ACQUIRE:
            return AcquireConfig.model_validate(_acquire_workflow_payload(payload))
        if workflow is WorkflowTask.TRAIN:
            return TrainConfig.model_validate(_model_workflow_payload(payload))
        if workflow is WorkflowTask.TUNE:
            return TuneConfig.model_validate(_model_workflow_payload(payload))
        if workflow is WorkflowTask.EVALUATE:
            return EvaluateConfig.model_validate(_evaluate_workflow_payload(payload))
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


def coerce_resolved_snapshot_workflow_config(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> ResolvedWorkflowConfig:
    if workflow is WorkflowTask.ACQUIRE:
        raise ConfigResolutionError(f"Unsupported resolved workflow: {workflow.value}")
    config = coerce_resolved_workflow_config(workflow, payload)
    if isinstance(config, (TrainConfig, TuneConfig, EvaluateConfig)):
        return config
    raise ConfigResolutionError(f"Unsupported resolved workflow: {workflow.value}")


def _acquire_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    return {
        **raw,
        "chain": _model_field(raw, "chain", ChainSpec),
        "dataset": _model_field(raw, "dataset", DatasetSpec),
        "storage": _model_field(raw, "storage", StorageSpec),
        "problem": coerce_problem_spec(_field(raw, "problem")),
        "features": coerce_features_config(_field(raw, "features")),
        "rpc_endpoint": _model_field(raw, "rpc_endpoint", ResolvedRpcEndpointConfig),
        "acquisition": _model_field(raw, "acquisition", AcquisitionConfig),
    }


def _evaluate_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    return {
        **raw,
        "storage": _model_field(raw, "storage", StorageSpec),
        "evaluation": coerce_evaluator_config(_field(raw, "evaluation")),
    }


def _model_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    problem = coerce_problem_spec(_field(raw, "problem"))
    model = coerce_model_config(_field(raw, "model"))
    tuning_space_payload = raw.get("tuning_space")
    tuning_space = (
        None
        if tuning_space_payload is None
        else coerce_tuning_space_config(
            _mapping_or_model(tuning_space_payload, label="tuning_space"),
            model_config=model,
            problem_config=problem,
        )
    )
    return {
        **raw,
        "chain": _model_field(raw, "chain", ChainSpec),
        "dataset": _model_field(raw, "dataset", DatasetSpec),
        "storage": _model_field(raw, "storage", StorageSpec),
        "problem": problem,
        "model": model,
        "dataset_builder": coerce_dataset_builder_config(_field(raw, "dataset_builder")),
        "features": coerce_features_config(_field(raw, "features")),
        "prediction": _model_field(raw, "prediction", PredictionConfig),
        "objective": coerce_objective_config(_field(raw, "objective")),
        "evaluation": _optional_evaluation(raw.get("evaluation")),
        "study": _model_field(raw, "study", StudyConfig),
        "artifact": _model_field(raw, "artifact", ArtifactConfig),
        "split": _model_field(raw, "split", SplitConfig),
        "training": _model_field(raw, "training", TrainingConfig),
        "tuning": _optional_tuning(raw.get("tuning")),
        "tuning_space": tuning_space,
    }


def _optional_evaluation(payload: object) -> EvaluatorConfig | None:
    if payload is None:
        return None
    if isinstance(payload, EvaluatorConfig) or isinstance(payload, Mapping):
        return coerce_evaluator_config(payload)
    raise ConfigResolutionError("resolved workflow config field evaluation must be a mapping")


def _optional_tuning(payload: object) -> TuningConfig | None:
    if payload is None:
        return None
    if isinstance(payload, TuningConfig):
        return payload
    return TuningConfig.model_validate(_mapping_or_model(payload, label="tuning"))


def _model_field(
    payload: Mapping[str, object],
    key: str,
    model_type: type[ConfigModelT],
) -> ConfigModelT:
    value = _field(payload, key)
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(_mapping_or_model(value, label=key))


def _field(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ConfigResolutionError(f"resolved workflow config field {key} is required")
    return payload[key]


def _mapping_or_model(payload: object, *, label: str) -> Mapping[str, object] | ConfigModel:
    if isinstance(payload, ConfigModel):
        return payload
    if isinstance(payload, Mapping):
        return payload
    raise ConfigResolutionError(f"resolved workflow config field {label} must be a mapping")

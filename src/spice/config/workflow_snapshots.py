"""Resolved Workflow Snapshot codec."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

from pydantic import ValidationError

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import ObjectiveConfig, coerce_objective_config
from .models import (
    ArtifactConfig,
    ChainSpec,
    DatasetBuilderConfig,
    DatasetSpec,
    EvaluateConfig,
    FeaturesConfig,
    ModelConfig,
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
    coerce_features_config,
    coerce_problem_spec,
)

ResolvedWorkflowConfig: TypeAlias = TrainConfig | TuneConfig | EvaluateConfig

ConfigModelT = TypeVar("ConfigModelT", bound=ConfigModel)
OwnerConfigT = TypeVar("OwnerConfigT")

_SNAPSHOT_WORKFLOWS = frozenset(
    {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}
)


def workflow_config_snapshot_payload(
    config: ResolvedWorkflowConfig,
    *,
    storage_root_override: Path | None = None,
) -> dict[str, object]:
    snapshot_config = config
    if storage_root_override is not None:
        snapshot_config = config.model_copy(
            update={"storage": StorageSpec(root=storage_root_override)}
        )
    return cast(
        dict[str, object],
        snapshot_config.model_dump(mode="json", exclude_none=True),
    )


def workflow_config_snapshot_json(
    config: ResolvedWorkflowConfig,
    *,
    storage_root_override: Path | None = None,
) -> str:
    return json.dumps(
        workflow_config_snapshot_payload(
            config,
            storage_root_override=storage_root_override,
        ),
        sort_keys=True,
    )


def hydrate_workflow_config_snapshot_json(
    workflow: WorkflowTask,
    payload: str,
) -> ResolvedWorkflowConfig:
    try:
        raw_payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    if not isinstance(raw_payload, Mapping):
        raise ConfigResolutionError("resolved workflow snapshot must be a mapping")
    return hydrate_workflow_config_snapshot(workflow, raw_payload)


def hydrate_workflow_config_snapshot(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> ResolvedWorkflowConfig:
    _validate_snapshot_workflow(workflow, payload)
    try:
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
    raise ConfigResolutionError(f"Unsupported resolved workflow: {workflow.value}")


def _validate_snapshot_workflow(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> None:
    if workflow not in _SNAPSHOT_WORKFLOWS:
        raise ConfigResolutionError(f"Unsupported resolved workflow: {workflow.value}")
    raw_workflow = payload.get("workflow")
    if raw_workflow is None:
        raise ConfigResolutionError("resolved workflow snapshot workflow is required")
    try:
        snapshot_workflow = WorkflowTask(str(raw_workflow))
    except ValueError as exc:
        raise ConfigResolutionError(
            f"resolved workflow snapshot has invalid workflow: {raw_workflow}"
        ) from exc
    if snapshot_workflow is not workflow:
        raise ConfigResolutionError(
            "resolved workflow snapshot workflow mismatch: "
            f"expected {workflow.value}, got {snapshot_workflow.value}"
        )


def _evaluate_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    return {
        **raw,
        "storage": _model_field(raw, "storage", StorageSpec),
        "evaluation": coerce_evaluator_config(
            _owner_field(raw, "evaluation", EvaluatorConfig)
        ),
    }


def _model_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    problem = coerce_problem_spec(_owner_field(raw, "problem", ProblemSpec))
    model = coerce_model_config(_model_config_field(raw, "model"))
    tuning_space_payload = raw.get("tuning_space")
    tuning_space = (
        None
        if tuning_space_payload is None
        else coerce_tuning_space_config(
            _mapping_or_owner(
                tuning_space_payload,
                label="tuning_space",
                config_type=TuningSpaceConfig,
            ),
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
        "dataset_builder": coerce_dataset_builder_config(
            _owner_field(raw, "dataset_builder", DatasetBuilderConfig)
        ),
        "features": coerce_features_config(_owner_field(raw, "features", FeaturesConfig)),
        "prediction": _model_field(raw, "prediction", PredictionConfig),
        "objective": coerce_objective_config(
            _owner_field(raw, "objective", ObjectiveConfig)
        ),
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
    return coerce_evaluator_config(
        _mapping_or_owner(payload, label="evaluation", config_type=EvaluatorConfig)
    )


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


def _owner_field(
    payload: Mapping[str, object],
    key: str,
    config_type: type[OwnerConfigT],
) -> Mapping[str, object] | OwnerConfigT:
    return _mapping_or_owner(_field(payload, key), label=key, config_type=config_type)


def _mapping_or_owner(
    payload: object,
    *,
    label: str,
    config_type: type[OwnerConfigT],
) -> Mapping[str, object] | OwnerConfigT:
    if isinstance(payload, config_type):
        return payload
    if isinstance(payload, Mapping):
        return payload
    raise ConfigResolutionError(f"resolved workflow config field {label} must be a mapping")


def _model_config_field(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | ModelConfig[str]:
    value = _field(payload, key)
    if isinstance(value, ModelConfig):
        return value
    if isinstance(value, Mapping):
        return value
    raise ConfigResolutionError(f"resolved workflow config field {key} must be a mapping")


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

"""Resolved Workflow Snapshot codec."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

from pydantic import ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, coerce_evaluator_config
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..objectives import coerce_objective_config
from .models import (
    ArtifactConfig,
    ChainSpec,
    DatasetSpec,
    EvaluateConfig,
    PredictionConfig,
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

ResolvedWorkflowConfig: TypeAlias = TrainConfig | TuneConfig | EvaluateConfig

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
    try:
        _validate_snapshot_workflow(workflow, payload)
        if workflow is WorkflowTask.EVALUATE:
            return EvaluateConfig.model_validate(_evaluate_workflow_payload(payload))
        resolved_payload = _model_workflow_payload(payload)
        if workflow is WorkflowTask.TRAIN:
            return TrainConfig.model_validate(resolved_payload)
        if workflow is WorkflowTask.TUNE:
            return TuneConfig.model_validate(resolved_payload)
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
        return
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
        "storage": StorageSpec.model_validate(_mapping_field(raw, "storage")),
        "evaluation": coerce_evaluator_config(_mapping_field(raw, "evaluation")),
    }


def _model_workflow_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw = dict(payload)
    problem = coerce_problem_spec(_mapping_field(raw, "problem"))
    model = coerce_model_config(_mapping_field(raw, "model"))
    tuning_space_payload = raw.get("tuning_space")
    tuning_space = (
        None
        if tuning_space_payload is None
        else coerce_tuning_space_config(
            _mapping_value(tuning_space_payload, label="tuning_space"),
            model_config=model,
            problem_config=problem,
        )
    )
    return {
        **raw,
        "chain": ChainSpec.model_validate(_mapping_field(raw, "chain")),
        "dataset": DatasetSpec.model_validate(_mapping_field(raw, "dataset")),
        "storage": StorageSpec.model_validate(_mapping_field(raw, "storage")),
        "problem": problem,
        "model": model,
        "dataset_builder": coerce_dataset_builder_config(_mapping_field(raw, "dataset_builder")),
        "features": coerce_features_config(_mapping_field(raw, "features")),
        "prediction": PredictionConfig.model_validate(_mapping_field(raw, "prediction")),
        "objective": coerce_objective_config(_mapping_field(raw, "objective")),
        "evaluation": _optional_evaluation(raw.get("evaluation")),
        "study": StudyConfig.model_validate(_mapping_field(raw, "study")),
        "artifact": ArtifactConfig.model_validate(_mapping_field(raw, "artifact")),
        "split": SplitConfig.model_validate(_mapping_field(raw, "split")),
        "training": TrainingConfig.model_validate(_mapping_field(raw, "training")),
        "tuning": _optional_tuning(raw.get("tuning")),
        "tuning_space": tuning_space,
    }


def _optional_evaluation(payload: object) -> EvaluatorConfig | None:
    if payload is None:
        return None
    return coerce_evaluator_config(_mapping_value(payload, label="evaluation"))


def _optional_tuning(payload: object) -> TuningConfig | None:
    if payload is None:
        return None
    return TuningConfig.model_validate(_mapping_value(payload, label="tuning"))


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _mapping_value(payload.get(key), label=key)


def _mapping_value(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError(f"resolved workflow snapshot field {label} must be a mapping")
    return payload

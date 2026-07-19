"""Resolved Workflow Snapshot codec."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from ..evaluation.config import EvaluatorConfig
from ..evaluation.registry import coerce_evaluator_config
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from .models import (
    ArtifactConfig,
    BlockWindowSpec,
    ChainSpec,
    CorpusSpec,
    FeaturesConfig,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TimestampWindowSpec,
    TrainingConfig,
    TuningConfig,
    TuningSpaceConfig,
    WorkflowTask,
    coerce_features_config,
    coerce_problem_spec,
)
from .resolved_workflows import (
    SUPPORTED_RESOLVED_WORKFLOWS,
    ResolvedEvaluateWorkflowFields,
    ResolvedModelWorkflowFields,
    ResolvedTrainWorkflowFields,
    ResolvedTuneWorkflowFields,
    ResolvedWorkflowConfig,
    assemble_resolved_workflow_config,
    final_evaluate_batch_size,
    resolved_workflow_field_names,
    resolved_workflow_snapshot_payload,
)

ConfigModelT = TypeVar("ConfigModelT", bound=ConfigModel)
OwnerConfigT = TypeVar("OwnerConfigT")

_SNAPSHOT_WORKFLOWS = frozenset(SUPPORTED_RESOLVED_WORKFLOWS)


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
    return resolved_workflow_snapshot_payload(snapshot_config)


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
    _validate_snapshot_fields(workflow, payload)
    try:
        if workflow is WorkflowTask.TRAIN:
            return assemble_resolved_workflow_config(_hydrate_train_fields(payload))
        if workflow is WorkflowTask.TUNE:
            return assemble_resolved_workflow_config(_hydrate_tune_fields(payload))
        if workflow is WorkflowTask.EVALUATE:
            return assemble_resolved_workflow_config(_hydrate_evaluate_fields(payload))
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


def _validate_snapshot_fields(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> None:
    allowed = resolved_workflow_field_names(workflow)
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ConfigResolutionError(
            "resolved workflow snapshot has unsupported fields: " + ", ".join(extra)
        )


def _hydrate_evaluate_fields(payload: Mapping[str, object]) -> ResolvedEvaluateWorkflowFields:
    raw = dict(payload)
    return ResolvedEvaluateWorkflowFields(
        storage=_model_field(raw, "storage", StorageSpec),
        artifact_id=_required_str(raw, "artifact_id"),
        corpus_id=_required_str(raw, "corpus_id"),
        evaluation_window=_model_field(
            raw,
            "evaluation_window",
            _evaluation_window_type(raw, "evaluation_window"),
        ),
        evaluator=coerce_evaluator_config(
            _owner_field(raw, "evaluator", EvaluatorConfig)
        ),
        delay_seconds=_optional_int(raw.get("delay_seconds"), label="delay_seconds"),
        batch_size=final_evaluate_batch_size(
            _optional_int(raw.get("batch_size"), label="batch_size")
        ),
    )


def _hydrate_train_fields(payload: Mapping[str, object]) -> ResolvedTrainWorkflowFields:
    raw = dict(payload)
    return ResolvedTrainWorkflowFields(
        model_fields=_hydrate_model_workflow_fields(raw),
        corpus_id=_optional_str(raw.get("corpus_id"), label="corpus_id"),
        study_id=_optional_str(raw.get("study_id"), label="study_id"),
        training_cutoff_timestamp=_optional_int(
            raw.get("training_cutoff_timestamp"),
            label="training_cutoff_timestamp",
        ),
    )


def _hydrate_tune_fields(payload: Mapping[str, object]) -> ResolvedTuneWorkflowFields:
    raw = dict(payload)
    return ResolvedTuneWorkflowFields(
        model_fields=_hydrate_model_workflow_fields(raw),
        corpus_id=_required_str(raw, "corpus_id"),
        training_cutoff_timestamp=_optional_int(
            raw.get("training_cutoff_timestamp"),
            label="training_cutoff_timestamp",
        ),
    )


def _hydrate_model_workflow_fields(
    raw: Mapping[str, object],
) -> ResolvedModelWorkflowFields:
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
    return ResolvedModelWorkflowFields(
        chain=_model_field(raw, "chain", ChainSpec),
        corpus=_model_field(raw, "corpus", CorpusSpec),
        storage=_model_field(raw, "storage", StorageSpec),
        problem=problem,
        model=model,
        features=coerce_features_config(_owner_field(raw, "features", FeaturesConfig)),
        prediction=_model_field(raw, "prediction", PredictionConfig),
        study=_model_field(raw, "study", StudyConfig),
        artifact=_model_field(raw, "artifact", ArtifactConfig),
        split=_model_field(raw, "split", SplitConfig),
        training=_model_field(raw, "training", TrainingConfig),
        tuning=_optional_tuning(raw.get("tuning")),
        tuning_space=tuning_space,
    )


def _optional_tuning(payload: object) -> TuningConfig | None:
    if payload is None:
        return None
    if isinstance(payload, TuningConfig):
        return payload
    return TuningConfig.model_validate(_mapping_or_model(payload, label="tuning"))


def _optional_str(payload: object, *, label: str) -> str | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise ConfigResolutionError(f"resolved workflow config field {label} must be a string")
    return payload


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = _field(payload, key)
    if not isinstance(value, str):
        raise ConfigResolutionError(f"resolved workflow config field {key} must be a string")
    return value


def _optional_int(payload: object, *, label: str) -> int | None:
    if payload is None:
        return None
    if isinstance(payload, bool):
        raise ConfigResolutionError(
            f"resolved workflow config field {label} must be an integer"
        )
    if not isinstance(payload, int):
        raise ConfigResolutionError(
            f"resolved workflow config field {label} must be an integer"
        )
    return payload


def _model_field(
    payload: Mapping[str, object],
    key: str,
    model_type: type[ConfigModelT],
) -> ConfigModelT:
    value = _field(payload, key)
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(_mapping_or_model(value, label=key))


def _evaluation_window_type(
    payload: Mapping[str, object],
    key: str,
) -> type[TimestampWindowSpec] | type[BlockWindowSpec]:
    value = _field(payload, key)
    if isinstance(value, BlockWindowSpec):
        return BlockWindowSpec
    if isinstance(value, TimestampWindowSpec):
        return TimestampWindowSpec
    mapping = _mapping_or_model(value, label=key)
    if isinstance(mapping, Mapping) and (
        "start_block" in mapping or "block_count" in mapping
    ):
        return BlockWindowSpec
    return TimestampWindowSpec


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

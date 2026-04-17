# pyright: strict

"""Explicit YAML config loading and workflow resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeVar, cast, overload

from pydantic import ValidationError

from ..core.errors import ConfigResolutionError
from ..modeling.families.registry import coerce_model_config, coerce_tuning_space_config
from .models import (
    AcquireConfig,
    AcquisitionConfig,
    ArtifactConfig,
    ChainSpec,
    ConfigModel,
    DatasetSpec,
    EvaluationConfig,
    EvaluateConfig,
    FeatureSetConfig,
    ModelConfig,
    PredictionConfig,
    PresetSpec,
    ProblemSpec,
    ProviderSpec,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
    WorkflowSelections,
    WorkflowTask,
    apply_provider_acquisition_overrides,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from .registry import list_group_names, load_named_group, named_group_keys

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"
_KNOWN_TOP_LEVEL_CONFIG_KEYS = frozenset(PresetSpec.model_fields)
_NAMED_GROUP_KEYS = frozenset(named_group_keys())

ModelT = TypeVar("ModelT", bound=ConfigModel)
WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


def _load_named_model(name: str) -> ModelConfig[str]:
    return coerce_model_config(load_named_group(name, _MODEL_GROUP))


def load_named_tuning_space(name: str, *, model_config: ModelConfig[str]) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        load_named_group(name, _TUNING_SPACE_GROUP),
        model_config=model_config,
    )
    if tuning_space is None:
        raise ConfigResolutionError(f"tuning space {name} resolved to None")
    return tuning_space


def _load_named_preset(name: str) -> PresetSpec:
    return PresetSpec.model_validate(load_named_group(name, "preset"))


def compact_mapping(payload: Mapping[str, object | None]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested = compact_mapping(
                _optional_mapping(cast(Mapping[object, object], value), label=key)
            )
            if nested:
                compacted[key] = nested
            continue
        compacted[key] = value
    return compacted


def _deep_merge_mappings(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            base_mapping = _mapping_copy(cast(Mapping[object, object], base_value), label=key)
            override_mapping = _mapping_copy(
                cast(Mapping[object, object], override_value),
                label=key,
            )
            if _replace_component_mapping(base_mapping, override_mapping):
                merged[key] = override_mapping
                continue
            merged[key] = _deep_merge_mappings(
                base_mapping,
                override_mapping,
            )
            continue
        merged[key] = _mapping_copy(
            cast(Mapping[object, object], override_value), label=key
        ) if isinstance(
            override_value, Mapping
        ) else override_value
    return merged


def _merge_workflow_payload(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(override_value, Mapping):
            if isinstance(base_value, Mapping):
                base_mapping = _mapping_copy(cast(Mapping[object, object], base_value), label=key)
                override_mapping = _mapping_copy(
                    cast(Mapping[object, object], override_value),
                    label=key,
                )
                if _replace_component_mapping(base_mapping, override_mapping):
                    merged[key] = override_mapping
                    continue
                merged[key] = _deep_merge_mappings(
                    base_mapping,
                    override_mapping,
                )
                continue
            if isinstance(base_value, str) and key in _NAMED_GROUP_KEYS:
                named_group = load_named_group(base_value, key)
                override_mapping = _mapping_copy(
                    cast(Mapping[object, object], override_value),
                    label=key,
                )
                if _replace_component_mapping(named_group, override_mapping):
                    merged[key] = override_mapping
                    continue
                merged[key] = _deep_merge_mappings(named_group, override_mapping)
                continue
            merged[key] = _mapping_copy(
                cast(Mapping[object, object], override_value), label=key
            )
            continue
        merged[key] = override_value
    return merged


def _replace_component_mapping(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> bool:
    base_id = base.get("id")
    override_id = override.get("id")
    return (
        isinstance(base_id, str)
        and isinstance(override_id, str)
        and base_id != override_id
    )


def _mapping_copy(value: Mapping[object, object], *, label: str) -> dict[str, object]:
    del label
    return {str(key): child for key, child in value.items()}


def _optional_mapping(
    value: Mapping[object, object],
    *,
    label: str,
) -> dict[str, object | None]:
    mapping = _mapping_copy(value, label=label)
    return {key: child for key, child in mapping.items()}


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.ACQUIRE],
    selections: WorkflowSelections,
) -> AcquireConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TRAIN],
    selections: WorkflowSelections,
) -> TrainConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.TUNE],
    selections: WorkflowSelections,
) -> TuneConfig: ...


@overload
def resolve_workflow_config(
    workflow_kind: Literal[WorkflowTask.EVALUATE],
    selections: WorkflowSelections,
) -> EvaluateConfig: ...


def resolve_workflow_config(
    workflow_kind: WorkflowTask | str,
    selections: WorkflowSelections,
) -> WorkflowConfig:
    """Resolve selector-driven workflow input into one validated workflow config."""

    try:
        workflow = (
            workflow_kind
            if isinstance(workflow_kind, WorkflowTask)
            else WorkflowTask(workflow_kind)
        )
        payload = _workflow_request(
            workflow=workflow,
            selections=selections,
        )
        if workflow is WorkflowTask.ACQUIRE:
            return _resolve_acquire_config(payload)
        if workflow is WorkflowTask.TRAIN:
            return _resolve_train_config(payload)
        if workflow is WorkflowTask.TUNE:
            return _resolve_tune_config(payload)
        if workflow is WorkflowTask.EVALUATE:
            return _resolve_evaluate_config(payload)
        raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc


def _workflow_request(
    *,
    workflow: WorkflowTask,
    selections: WorkflowSelections,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if selections.preset is not None:
        merged = _load_named_preset(selections.preset).model_dump(mode="json", exclude_none=True)
    merged = _merge_workflow_payload(
        merged,
        compact_mapping(
            {
                "dataset": selections.dataset,
                "problem": selections.problem,
                "chain": selections.chain,
                "provider": selections.provider if workflow is WorkflowTask.ACQUIRE else None,
                "model": selections.model
                if workflow in {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}
                else None,
                "feature_set": selections.feature_set,
                "prediction": selections.prediction
                if workflow in {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}
                else None,
                "study": selections.study
                if workflow in {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}
                else None,
                "artifact": {"variant": selections.variant}
                if selections.variant is not None
                else None,
                "storage": {"root": selections.storage_root}
                if selections.storage_root is not None
                else None,
                "acquisition": {"dry_run": selections.dry_run}
                if workflow is WorkflowTask.ACQUIRE and selections.dry_run is not None
                else None,
                "tuning": {"trial_count": selections.trial_count}
                if workflow is WorkflowTask.TUNE and selections.trial_count is not None
                else None,
                "delay_seconds": selections.delay_seconds
                if workflow is WorkflowTask.EVALUATE
                else None,
            }
        ),
    )
    _reject_unknown_top_level_keys(merged)
    return merged


def _reject_unknown_top_level_keys(payload: Mapping[str, object]) -> None:
    unknown = sorted(set(payload) - _KNOWN_TOP_LEVEL_CONFIG_KEYS)
    if unknown:
        raise ConfigResolutionError(f"Unknown top-level config fields: {', '.join(unknown)}")


def _require_payload_key(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ConfigResolutionError(f"Missing required workflow config field: {key}")
    return payload[key]


def resolve_named_or_inline(raw: object, *, group: str, model_type: type[ModelT]) -> ModelT:
    if isinstance(raw, str):
        return model_type.model_validate(load_named_group(raw, group))
    if isinstance(raw, Mapping):
        return model_type.model_validate(
            _mapping_copy(cast(Mapping[object, object], raw), label=group)
        )
    raise ConfigResolutionError(f"{group} must be provided as a spec name or mapping")


def resolve_inline(raw: object, *, label: str, model_type: type[ModelT]) -> ModelT:
    if isinstance(raw, Mapping):
        return model_type.model_validate(
            _mapping_copy(cast(Mapping[object, object], raw), label=label)
        )
    raise ConfigResolutionError(f"{label} must be provided as a mapping")


def resolve_problem(raw: object) -> ProblemSpec:
    if isinstance(raw, str):
        return coerce_problem_spec(load_named_group(raw, "problem"))
    if isinstance(raw, Mapping):
        return coerce_problem_spec(
            _mapping_copy(cast(Mapping[object, object], raw), label="problem")
        )
    raise ConfigResolutionError("problem must be provided as a spec name or mapping")


def resolve_feature_set(raw: object) -> FeatureSetConfig:
    if isinstance(raw, str):
        return coerce_feature_set_config(load_named_group(raw, "feature_set"))
    if isinstance(raw, Mapping):
        return coerce_feature_set_config(
            _mapping_copy(cast(Mapping[object, object], raw), label="feature_set")
        )
    raise ConfigResolutionError("feature_set must be provided as a spec name or mapping")


def resolve_prediction(raw: object) -> PredictionConfig:
    if isinstance(raw, str):
        return coerce_prediction_config(load_named_group(raw, "prediction"))
    if isinstance(raw, Mapping):
        return coerce_prediction_config(
            _mapping_copy(cast(Mapping[object, object], raw), label="prediction")
        )
    raise ConfigResolutionError("prediction must be provided as a spec name or mapping")


def resolve_storage(raw: object | None) -> StorageSpec:
    if raw is None:
        return StorageSpec()
    if isinstance(raw, (str, Path)):
        return StorageSpec(root=Path(raw))
    if isinstance(raw, Mapping):
        return StorageSpec.model_validate(
            _mapping_copy(cast(Mapping[object, object], raw), label="storage")
        )
    raise ConfigResolutionError("storage must be a path or mapping")


def _resolve_common(
    payload: dict[str, object],
) -> tuple[DatasetSpec, ChainSpec, StorageSpec]:
    dataset = resolve_named_or_inline(payload["dataset"], group="dataset", model_type=DatasetSpec)
    chain = resolve_named_or_inline(payload["chain"], group="chain", model_type=ChainSpec)
    storage = resolve_storage(payload.get("storage"))
    return dataset, chain, storage


def _resolve_provider(payload: dict[str, object], *, chain: ChainSpec) -> ProviderSpec:
    provider = resolve_named_or_inline(
        payload["provider"],
        group="provider",
        model_type=ProviderSpec,
    )
    unknown_chains = sorted(set(provider.chains) - set(list_group_names("chain")))
    if unknown_chains:
        raise ConfigResolutionError(
            f"provider {provider.name} declares unknown chains: {', '.join(unknown_chains)}"
        )
    provider.endpoint_for(chain.name)
    return provider


def _resolve_model_workflow(
    payload: dict[str, object],
) -> tuple[
    DatasetSpec,
    ChainSpec,
    StorageSpec,
    ProblemSpec,
    ModelConfig[str],
    FeatureSetConfig,
    PredictionConfig,
    StudyConfig,
    ArtifactConfig,
]:
    dataset, chain, storage = _resolve_common(payload)
    problem = resolve_problem(payload["problem"])
    model_raw = payload["model"]
    if isinstance(model_raw, str):
        model = _load_named_model(model_raw)
    elif isinstance(model_raw, Mapping):
        model = coerce_model_config(
            _mapping_copy(cast(Mapping[object, object], model_raw), label="model")
        )
    else:
        raise ConfigResolutionError("model must be provided as a spec name or mapping")
    feature_set = resolve_feature_set(payload["feature_set"])
    prediction = resolve_prediction(payload["prediction"])
    study_raw = payload.get("study")
    if isinstance(study_raw, Mapping):
        study = StudyConfig.model_validate(study_raw)
    elif isinstance(study_raw, str):
        study = StudyConfig(name=study_raw)
    else:
        study = StudyConfig()
    artifact_raw = payload.get("artifact")
    artifact = (
        ArtifactConfig.model_validate(artifact_raw)
        if isinstance(artifact_raw, Mapping)
        else ArtifactConfig()
    )
    return dataset, chain, storage, problem, model, feature_set, prediction, study, artifact


def _resolve_acquire_config(payload: dict[str, object]) -> AcquireConfig:
    dataset_spec, chain_spec, storage_spec = _resolve_common(payload)
    problem_spec = resolve_problem(payload["problem"])
    provider_spec = _resolve_provider(payload, chain=chain_spec)
    feature_set_spec = resolve_feature_set(payload["feature_set"])
    acquisition_spec = resolve_inline(
        _require_payload_key(payload, "acquisition"),
        label="acquisition",
        model_type=AcquisitionConfig,
    )
    acquisition_spec = apply_provider_acquisition_overrides(
        provider=provider_spec,
        acquisition=acquisition_spec,
    )
    return AcquireConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        problem=problem_spec,
        feature_set=feature_set_spec,
        provider=provider_spec,
        acquisition=acquisition_spec,
    )


def _resolve_train_config(payload: dict[str, object]) -> TrainConfig:
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        problem_spec,
        model_spec,
        feature_set_spec,
        prediction_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_inline(
        _require_payload_key(payload, "training"),
        label="training",
        model_type=TrainingConfig,
    )
    split_spec = resolve_inline(
        _require_payload_key(payload, "split"),
        label="split",
        model_type=SplitConfig,
    )
    return TrainConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        problem=problem_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        prediction=prediction_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        split=split_spec,
    )


def _resolve_tune_config(payload: dict[str, object]) -> TuneConfig:
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        problem_spec,
        model_spec,
        feature_set_spec,
        prediction_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_inline(
        _require_payload_key(payload, "training"),
        label="training",
        model_type=TrainingConfig,
    )
    split_spec = resolve_inline(
        _require_payload_key(payload, "split"),
        label="split",
        model_type=SplitConfig,
    )
    tuning_raw = _require_payload_key(payload, "tuning")
    tuning_spec = resolve_inline(
        tuning_raw,
        label="tuning",
        model_type=TuningConfig,
    )
    tuning_space_raw = _require_payload_key(payload, "tuning_space")
    if isinstance(tuning_space_raw, str):
        tuning_space_spec = load_named_tuning_space(tuning_space_raw, model_config=model_spec)
    elif isinstance(tuning_space_raw, Mapping):
        resolved_tuning_space = coerce_tuning_space_config(
            _mapping_copy(
                cast(Mapping[object, object], tuning_space_raw), label="tuning_space"
            ),
            model_config=model_spec,
        )
        if resolved_tuning_space is None:
            raise ConfigResolutionError("tuning_space is required for tune")
        tuning_space_spec = resolved_tuning_space
    else:
        raise ConfigResolutionError("tuning_space must be provided as a spec name or mapping")
    return TuneConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        problem=problem_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        prediction=prediction_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        split=split_spec,
        tuning=tuning_spec,
        tuning_space=tuning_space_spec,
    )


def _resolve_evaluate_config(payload: dict[str, object]) -> EvaluateConfig:
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        problem_spec,
        model_spec,
        feature_set_spec,
        prediction_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_inline(
        _require_payload_key(payload, "training"),
        label="training",
        model_type=TrainingConfig,
    )
    evaluation_spec = resolve_inline(
        _require_payload_key(payload, "evaluation"),
        label="evaluation",
        model_type=EvaluationConfig,
    )
    delay_raw = _require_payload_key(payload, "delay_seconds")
    if not isinstance(delay_raw, int):
        raise ConfigResolutionError("delay_seconds must be an integer")
    delay_seconds = delay_raw
    return EvaluateConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        problem=problem_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        prediction=prediction_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        evaluation=evaluation_spec,
        delay_seconds=delay_seconds,
    )

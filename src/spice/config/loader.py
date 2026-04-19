# pyright: strict

"""Explicit YAML config loading and workflow resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    DatasetBuilderConfig,
    DatasetSpec,
    EvaluateConfig,
    EvaluationConfig,
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
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from .registry import list_group_names, load_named_group

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"
_KNOWN_TOP_LEVEL_CONFIG_KEYS = frozenset(PresetSpec.model_fields)

ModelT = TypeVar("ModelT", bound=ConfigModel)
WorkflowConfig = AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig


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


def _mapping_copy(value: Mapping[object, object]) -> dict[str, object]:
    return {str(key): child for key, child in value.items()}


def _require_mapping(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ConfigResolutionError(f"{label} must be provided as a mapping")
    return _mapping_copy(cast(Mapping[object, object], raw))


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
    payload = (
        _load_named_preset(selections.preset).model_dump(mode="json", exclude_none=True)
        if selections.preset is not None
        else {}
    )
    _apply_selection_overlays(payload, workflow=workflow, selections=selections)
    _reject_unknown_top_level_keys(payload)
    return payload


def _apply_selection_overlays(
    payload: dict[str, object],
    *,
    workflow: WorkflowTask,
    selections: WorkflowSelections,
) -> None:
    for field in ("dataset", "problem", "chain", "feature_set"):
        value = getattr(selections, field)
        if value is not None:
            payload[field] = value
    if selections.storage_root is not None:
        payload["storage"] = _overlay_mapping(
            payload.get("storage"),
            {"root": selections.storage_root},
            label="storage",
        )
    if workflow is WorkflowTask.ACQUIRE:
        if selections.provider is not None:
            payload["provider"] = selections.provider
        if selections.dry_run is not None:
            payload["acquisition"] = _overlay_mapping(
                payload.get("acquisition"),
                {"dry_run": selections.dry_run},
                label="acquisition",
            )
        return
    for field in ("model", "dataset_builder", "prediction", "study"):
        value = getattr(selections, field)
        if value is not None:
            payload[field] = value
    if selections.variant is not None:
        payload["artifact"] = _overlay_mapping(
            payload.get("artifact"),
            {"variant": selections.variant},
            label="artifact",
        )
    if workflow is WorkflowTask.TUNE and selections.trial_count is not None:
        payload["tuning"] = _overlay_mapping(
            payload.get("tuning"),
            {"trial_count": selections.trial_count},
            label="tuning",
        )
    if workflow is WorkflowTask.EVALUATE and selections.delay_seconds is not None:
        payload["delay_seconds"] = selections.delay_seconds


def _overlay_mapping(
    current: object,
    overlay: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if current is None:
        return dict(overlay)
    if not isinstance(current, Mapping):
        raise ConfigResolutionError(f"{label} must be provided as a mapping")
    return {
        **_mapping_copy(cast(Mapping[object, object], current)),
        **overlay,
    }


def _reject_unknown_top_level_keys(payload: Mapping[str, object]) -> None:
    unknown = sorted(set(payload) - _KNOWN_TOP_LEVEL_CONFIG_KEYS)
    if unknown:
        raise ConfigResolutionError(f"Unknown top-level config fields: {', '.join(unknown)}")


def _require_payload_key(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ConfigResolutionError(f"Missing required workflow config field: {key}")
    return payload[key]


def resolve_named_or_inline(raw: object, *, group: str, model_type: type[ModelT]) -> ModelT:
    return _resolve_named_or_mapping(
        raw,
        group=group,
        label=group,
        parse_mapping=model_type.model_validate,
    )


def _resolve_named_or_mapping(
    raw: object,
    *,
    group: str,
    label: str,
    parse_mapping: Callable[[dict[str, object]], ModelT],
) -> ModelT:
    if isinstance(raw, str):
        return parse_mapping(load_named_group(raw, group))
    if isinstance(raw, Mapping):
        return parse_mapping(_mapping_copy(cast(Mapping[object, object], raw)))
    raise ConfigResolutionError(f"{label} must be provided as a spec name or mapping")


def resolve_inline(raw: object, *, label: str, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(_require_mapping(raw, label=label))


def resolve_problem(raw: object) -> ProblemSpec:
    return _resolve_named_or_mapping(
        raw,
        group="problem",
        label="problem",
        parse_mapping=coerce_problem_spec,
    )


def resolve_feature_set(raw: object) -> FeatureSetConfig:
    return _resolve_named_or_mapping(
        raw,
        group="feature_set",
        label="feature_set",
        parse_mapping=coerce_feature_set_config,
    )


def resolve_dataset_builder(raw: object) -> DatasetBuilderConfig:
    return _resolve_named_or_mapping(
        raw,
        group="dataset_builder",
        label="dataset_builder",
        parse_mapping=coerce_dataset_builder_config,
    )


def resolve_prediction(raw: object) -> PredictionConfig:
    return _resolve_named_or_mapping(
        raw,
        group="prediction",
        label="prediction",
        parse_mapping=coerce_prediction_config,
    )


def _resolve_model(raw: object) -> ModelConfig[str]:
    return _resolve_named_or_mapping(
        raw,
        group=_MODEL_GROUP,
        label="model",
        parse_mapping=coerce_model_config,
    )


def _resolve_tuning_space(
    raw: object,
    *,
    model_config: ModelConfig[str],
) -> TuningSpaceConfig:
    if isinstance(raw, str):
        return load_named_tuning_space(raw, model_config=model_config)
    if isinstance(raw, Mapping):
        resolved = coerce_tuning_space_config(
            _mapping_copy(cast(Mapping[object, object], raw)),
            model_config=model_config,
        )
        if resolved is None:
            raise ConfigResolutionError("tuning_space is required for tune")
        return resolved
    raise ConfigResolutionError("tuning_space must be provided as a spec name or mapping")


def resolve_storage(raw: object | None) -> StorageSpec:
    if raw is None:
        return StorageSpec()
    if isinstance(raw, (str, Path)):
        return StorageSpec(root=Path(raw))
    return StorageSpec.model_validate(_require_mapping(raw, label="storage"))


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
    DatasetBuilderConfig,
    FeatureSetConfig,
    PredictionConfig,
    StudyConfig,
    ArtifactConfig,
]:
    dataset, chain, storage = _resolve_common(payload)
    problem = resolve_problem(payload["problem"])
    model = _resolve_model(payload["model"])
    dataset_builder = resolve_dataset_builder(_require_payload_key(payload, "dataset_builder"))
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
    return (
        dataset,
        chain,
        storage,
        problem,
        model,
        dataset_builder,
        feature_set,
        prediction,
        study,
        artifact,
    )


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
        dataset_builder_spec,
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
        dataset_builder=dataset_builder_spec,
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
        dataset_builder_spec,
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
    tuning_space_spec = _resolve_tuning_space(
        _require_payload_key(payload, "tuning_space"),
        model_config=model_spec,
    )
    return TuneConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        problem=problem_spec,
        model=model_spec,
        dataset_builder=dataset_builder_spec,
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
        dataset_builder_spec,
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
        dataset_builder=dataset_builder_spec,
        feature_set=feature_set_spec,
        prediction=prediction_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        evaluation=evaluation_spec,
        delay_seconds=delay_seconds,
    )

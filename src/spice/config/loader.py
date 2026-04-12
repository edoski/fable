"""Explicit YAML config loading and composition."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import TypeVar, cast

import yaml

from ..modeling.registry import coerce_model_config, coerce_tuning_space_config
from .models import (
    AcquireConfig,
    AcquisitionConfig,
    ArtifactConfig,
    ChainSpec,
    ConfigModel,
    DatasetSpec,
    FeatureSetConfig,
    ModelConfig,
    PresetSpec,
    ProviderSpec,
    SimulateConfig,
    SimulationConfig,
    SplitConfig,
    StorageSpec,
    StudyConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
    apply_provider_acquisition_overrides,
)

_MODEL_GROUP = "model"
_TUNING_SPACE_GROUP = "tuning_space"
_CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
_MERGEABLE_NAMED_GROUPS = {
    "dataset": "dataset",
    "chain": "chain",
    "provider": "provider",
    "feature_set": "feature_set",
    "training": "training",
    "split": "split",
    "simulation": "simulation",
    "acquisition": "acquisition",
    "tuning": "tuning",
    "model": _MODEL_GROUP,
    "tuning_space": _TUNING_SPACE_GROUP,
}

ModelT = TypeVar("ModelT", bound=ConfigModel)


def load_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return cast(dict[str, object], payload)


def load_named_group(name: str, group: str) -> dict[str, object]:
    path = _CONF_ROOT / group / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {group} spec: {name}")
    return load_yaml_mapping(path)


@cache
def _available_group_names(group: str) -> frozenset[str]:
    directory = _CONF_ROOT / group
    return frozenset(path.stem for path in directory.glob("*.yaml"))


def load_named_model(name: str) -> ModelConfig:
    return coerce_model_config(load_named_group(name, _MODEL_GROUP))


def load_named_tuning_space(name: str, *, model_config: ModelConfig) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        load_named_group(name, _TUNING_SPACE_GROUP),
        model_config=model_config,
    )
    if tuning_space is None:
        raise ValueError(f"tuning space {name} resolved to None")
    return tuning_space


def load_named_preset(name: str) -> PresetSpec:
    return PresetSpec.model_validate(load_named_group(name, "preset"))


def deep_merge(base: dict[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if (
            isinstance(existing, str)
            and isinstance(value, Mapping)
            and key in _MERGEABLE_NAMED_GROUPS
        ):
            merged[key] = deep_merge(
                load_named_group(existing, _MERGEABLE_NAMED_GROUPS[key]),
                cast(Mapping[str, object], value),
            )
        elif isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, cast(Mapping[str, object], value))
        else:
            merged[key] = cast(object, value)
    return merged


def read_config_override(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return load_yaml_mapping(path)


def compact_mapping(payload: Mapping[str, object | None]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested = compact_mapping(cast(Mapping[str, object | None], value))
            if nested:
                compacted[key] = nested
            continue
        compacted[key] = value
    return compacted


def resolve_named_or_inline(raw: object, *, group: str, model_type: type[ModelT]) -> ModelT:
    if isinstance(raw, str):
        return model_type.model_validate(load_named_group(raw, group))
    if isinstance(raw, Mapping):
        return model_type.model_validate(dict(raw))
    raise ValueError(f"{group} must be provided as a spec name or mapping")


def resolve_storage(raw: object | None) -> StorageSpec:
    if raw is None:
        return StorageSpec()
    if isinstance(raw, (str, Path)):
        return StorageSpec(root=Path(raw))
    if isinstance(raw, Mapping):
        return StorageSpec.model_validate(dict(raw))
    raise ValueError("storage must be a path or mapping")


def _merged_request(
    *,
    preset_name: str | None,
    config_path: Path | None,
    cli_overrides: dict[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if preset_name is not None:
        merged = deep_merge(merged, load_named_preset(preset_name).model_dump(mode="json"))
    config_override = read_config_override(config_path)
    preset_override = config_override.get("preset")
    if isinstance(preset_override, str):
        merged = deep_merge(merged, load_named_preset(preset_override).model_dump(mode="json"))
        config_override = {key: value for key, value in config_override.items() if key != "preset"}
    return deep_merge(deep_merge(merged, config_override), cli_overrides)


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
    unknown_chains = sorted(set(provider.chains) - set(_available_group_names("chain")))
    if unknown_chains:
        raise ValueError(
            f"provider {provider.name} declares unknown chains: {', '.join(unknown_chains)}"
        )
    provider.endpoint_for(chain.name)
    return provider


def load_acquire_config(
    *,
    preset: str | None = None,
    config_path: Path | None = None,
    dataset: str | None = None,
    chain: str | None = None,
    provider: str | None = None,
    acquisition: str | None = None,
    storage_root: Path | None = None,
    dry_run: bool | None = None,
) -> AcquireConfig:
    payload = _merged_request(
        preset_name=preset,
        config_path=config_path,
        cli_overrides=compact_mapping(
            {
                "dataset": dataset,
                "chain": chain,
                "provider": provider,
                "acquisition": {"dry_run": dry_run} if dry_run is not None else None,
                "storage": {"root": storage_root} if storage_root is not None else None,
            }
        ),
    )
    dataset_spec, chain_spec, storage_spec = _resolve_common(payload)
    provider_spec = _resolve_provider(payload, chain=chain_spec)
    acquisition_raw = payload.get("acquisition", "default")
    acquisition_spec = resolve_named_or_inline(
        acquisition_raw,
        group="acquisition",
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
        provider=provider_spec,
        acquisition=acquisition_spec,
    )


def _resolve_model_workflow(
    payload: dict[str, object],
) -> tuple[
    DatasetSpec,
    ChainSpec,
    StorageSpec,
    ModelConfig,
    FeatureSetConfig,
    StudyConfig,
    ArtifactConfig,
]:
    dataset, chain, storage = _resolve_common(payload)
    model_raw = payload["model"]
    if isinstance(model_raw, str):
        model = load_named_model(model_raw)
    elif isinstance(model_raw, Mapping):
        model = coerce_model_config(dict(model_raw))
    else:
        raise ValueError("model must be provided as a spec name or mapping")
    feature_set = resolve_named_or_inline(
        payload["feature_set"],
        group="feature_set",
        model_type=FeatureSetConfig,
    )
    study_raw = payload.get("study")
    if isinstance(study_raw, Mapping):
        study = StudyConfig.model_validate(study_raw)
    else:
        study = StudyConfig(id=cast(str, study_raw) if isinstance(study_raw, str) else "default")
    artifact_raw = payload.get("artifact")
    artifact = (
        ArtifactConfig.model_validate(artifact_raw)
        if isinstance(artifact_raw, Mapping)
        else ArtifactConfig()
    )
    return dataset, chain, storage, model, feature_set, study, artifact


def load_train_config(
    *,
    preset: str | None = None,
    config_path: Path | None = None,
    dataset: str | None = None,
    chain: str | None = None,
    model: str | None = None,
    feature_set: str | None = None,
    training: str | None = None,
    split: str | None = None,
    storage_root: Path | None = None,
    variant: str | None = None,
    study: str | None = None,
) -> TrainConfig:
    payload = _merged_request(
        preset_name=preset,
        config_path=config_path,
        cli_overrides=compact_mapping(
            {
                "dataset": dataset,
                "chain": chain,
                "model": model,
                "feature_set": feature_set,
                "training": training,
                "split": split,
                "study": study,
                "artifact": {"variant": variant} if variant is not None else None,
                "storage": {"root": storage_root} if storage_root is not None else None,
            }
        ),
    )
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        model_spec,
        feature_set_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_named_or_inline(
        payload.get("training", "default"),
        group="training",
        model_type=TrainingConfig,
    )
    split_spec = resolve_named_or_inline(
        payload.get("split", "default"),
        group="split",
        model_type=SplitConfig,
    )
    return TrainConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        split=split_spec,
    )


def load_tune_config(
    *,
    preset: str | None = None,
    config_path: Path | None = None,
    dataset: str | None = None,
    chain: str | None = None,
    model: str | None = None,
    feature_set: str | None = None,
    training: str | None = None,
    split: str | None = None,
    tuning: str | None = None,
    tuning_space: str | None = None,
    storage_root: Path | None = None,
    study: str | None = None,
    trial_count: int | None = None,
) -> TuneConfig:
    payload = _merged_request(
        preset_name=preset,
        config_path=config_path,
        cli_overrides=compact_mapping(
            {
                "dataset": dataset,
                "chain": chain,
                "model": model,
                "feature_set": feature_set,
                "training": training,
                "split": split,
                "tuning": {"trial_count": trial_count} if trial_count is not None else tuning,
                "tuning_space": tuning_space,
                "study": study,
                "storage": {"root": storage_root} if storage_root is not None else None,
            }
        ),
    )
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        model_spec,
        feature_set_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_named_or_inline(
        payload.get("training", "default"),
        group="training",
        model_type=TrainingConfig,
    )
    split_spec = resolve_named_or_inline(
        payload.get("split", "default"),
        group="split",
        model_type=SplitConfig,
    )
    tuning_raw = payload.get("tuning", "default")
    tuning_spec = resolve_named_or_inline(
        tuning_raw,
        group="tuning",
        model_type=TuningConfig,
    )
    tuning_space_raw = payload.get("tuning_space")
    if tuning_space_raw is None:
        default_name = f"{model_spec.id}_default"
        tuning_space_spec = load_named_tuning_space(default_name, model_config=model_spec)
    elif isinstance(tuning_space_raw, str):
        tuning_space_spec = load_named_tuning_space(tuning_space_raw, model_config=model_spec)
    elif isinstance(tuning_space_raw, Mapping):
        tuning_space = coerce_tuning_space_config(dict(tuning_space_raw), model_config=model_spec)
        if tuning_space is None:
            raise ValueError("tuning_space is required for tune")
        tuning_space_spec = tuning_space
    else:
        raise ValueError("tuning_space must be provided as a spec name or mapping")
    return TuneConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        split=split_spec,
        tuning=tuning_spec,
        tuning_space=tuning_space_spec,
    )


def load_simulate_config(
    *,
    preset: str | None = None,
    config_path: Path | None = None,
    dataset: str | None = None,
    chain: str | None = None,
    model: str | None = None,
    feature_set: str | None = None,
    training: str | None = None,
    simulation: str | None = None,
    storage_root: Path | None = None,
    variant: str | None = None,
    study: str | None = None,
) -> SimulateConfig:
    payload = _merged_request(
        preset_name=preset,
        config_path=config_path,
        cli_overrides=compact_mapping(
            {
                "dataset": dataset,
                "chain": chain,
                "model": model,
                "feature_set": feature_set,
                "training": training,
                "simulation": simulation,
                "study": study,
                "artifact": {"variant": variant} if variant is not None else None,
                "storage": {"root": storage_root} if storage_root is not None else None,
            }
        ),
    )
    (
        dataset_spec,
        chain_spec,
        storage_spec,
        model_spec,
        feature_set_spec,
        study_spec,
        artifact_spec,
    ) = _resolve_model_workflow(payload)
    training_spec = resolve_named_or_inline(
        payload.get("training", "default"),
        group="training",
        model_type=TrainingConfig,
    )
    simulation_spec = resolve_named_or_inline(
        payload.get("simulation", "default"),
        group="simulation",
        model_type=SimulationConfig,
    )
    return SimulateConfig(
        chain=chain_spec,
        dataset=dataset_spec,
        storage=storage_spec,
        model=model_spec,
        feature_set=feature_set_spec,
        study=study_spec,
        artifact=artifact_spec,
        training=training_spec,
        simulation=simulation_spec,
    )

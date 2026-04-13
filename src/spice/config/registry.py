"""Config spec registry and authoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel

from .models import (
    ChainSpec,
    ConfigModel,
    DatasetSpec,
    ExecutionSpec,
    FeatureSetConfig,
    PredictionConfig,
    PresetSpec,
    ProblemSpec,
    ProviderSpec,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)

_CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"


class ConfigGroup(StrEnum):
    CHAIN = "chain"
    PROVIDER = "provider"
    DATASET = "dataset"
    PROBLEM = "problem"
    EXECUTION = "execution"
    FEATURE_SET = "feature-set"
    PREDICTION = "prediction"
    PRESET = "preset"


@dataclass(frozen=True, slots=True)
class ConfigGroupDefinition:
    token: str
    directory: str
    model_type: type[ConfigModel]
    identity_field: str | None = None
    authorable: bool = False


_GROUP_DEFINITIONS = (
    ConfigGroupDefinition(
        token=ConfigGroup.CHAIN.value,
        directory="chain",
        model_type=ChainSpec,
        identity_field="name",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.PROVIDER.value,
        directory="provider",
        model_type=ProviderSpec,
        identity_field="name",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.DATASET.value,
        directory="dataset",
        model_type=DatasetSpec,
        identity_field="name",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.PROBLEM.value,
        directory="problem",
        model_type=ProblemSpec,
        identity_field="id",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.EXECUTION.value,
        directory="execution",
        model_type=ExecutionSpec,
        identity_field="id",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.FEATURE_SET.value,
        directory="feature_set",
        model_type=FeatureSetConfig,
        identity_field="id",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.PREDICTION.value,
        directory="prediction",
        model_type=PredictionConfig,
        identity_field="id",
        authorable=True,
    ),
    ConfigGroupDefinition(
        token=ConfigGroup.PRESET.value,
        directory="preset",
        model_type=PresetSpec,
        authorable=True,
    ),
)
_GROUP_BY_TOKEN = {definition.token: definition for definition in _GROUP_DEFINITIONS}
_GROUP_BY_DIRECTORY = {definition.directory: definition for definition in _GROUP_DEFINITIONS}
_KNOWN_GROUP_DIRECTORIES = frozenset(
    {
        "acquisition",
        "chain",
        "dataset",
        "execution",
        "feature_set",
        "prediction",
        "model",
        "preset",
        "provider",
        "simulation",
        "split",
        "problem",
        "training",
        "tuning",
        "tuning_space",
    }
)
_PRESET_REFERENCE_GROUPS = {
    "dataset": "dataset",
    "problem": "problem",
    "execution": "execution",
    "chain": "chain",
    "provider": "provider",
    "model": "model",
    "feature_set": "feature_set",
    "prediction": "prediction",
    "acquisition": "acquisition",
    "training": "training",
    "split": "split",
    "simulation": "simulation",
    "tuning": "tuning",
    "tuning_space": "tuning_space",
}


def config_root() -> Path:
    return _CONF_ROOT


def authorable_group_tokens() -> tuple[str, ...]:
    return tuple(definition.token for definition in _GROUP_DEFINITIONS if definition.authorable)


def normalize_group_name(group: str) -> str:
    if group in _GROUP_BY_TOKEN:
        return _GROUP_BY_TOKEN[group].directory
    if group in _GROUP_BY_DIRECTORY or group in _KNOWN_GROUP_DIRECTORIES:
        return group
    raise ValueError(f"Unsupported config group: {group}")


def group_path(group: str) -> Path:
    return config_root() / normalize_group_name(group)


def spec_path(group: str, name: str) -> Path:
    return group_path(group) / f"{name}.yaml"


def load_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return dict(payload)


def load_named_group(name: str, group: str) -> dict[str, object]:
    path = spec_path(group, name)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {normalize_group_name(group)} spec: {name}")
    return load_yaml_mapping(path)


def list_group_names(group: str) -> list[str]:
    directory = group_path(group)
    return sorted(path.stem for path in directory.glob("*.yaml"))


def show_named_group(group_token: str, name: str) -> str:
    definition = authorable_group_definition(group_token)
    model = validated_model_from_disk(name=name, definition=definition)
    return dump_canonical_yaml(model)


def create_named_group(
    *,
    group_token: str,
    name: str,
    set_values: list[str],
) -> Path:
    definition = authorable_group_definition(group_token)
    path = spec_path(definition.directory, name)
    if path.exists():
        raise FileExistsError(f"{definition.token} spec already exists: {name}")
    payload = _seed_payload(definition, name)
    _apply_set_operations(payload, set_values)
    model = validate_mapping_for_write(definition=definition, name=name, payload=payload)
    return write_named_group(definition=definition, name=name, model=model)


def update_named_group(
    *,
    group_token: str,
    name: str,
    set_values: list[str],
    unset_paths: list[str],
) -> Path:
    definition = authorable_group_definition(group_token)
    if not set_values and not unset_paths:
        raise ValueError("update requires at least one --set or --unset")
    path = spec_path(definition.directory, name)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {definition.token} spec: {name}")
    payload = load_yaml_mapping(path)
    _apply_set_operations(payload, set_values)
    _apply_unset_operations(payload, unset_paths)
    model = validate_mapping_for_write(definition=definition, name=name, payload=payload)
    return write_named_group(definition=definition, name=name, model=model)


def delete_named_group(
    *,
    group_token: str,
    name: str,
    force: bool,
) -> list[str]:
    definition = authorable_group_definition(group_token)
    path = spec_path(definition.directory, name)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {definition.token} spec: {name}")
    dependents = dependent_specs(group_token=group_token, name=name)
    if dependents and not force:
        raise ValueError(
            "\n".join(
                [
                    f"Cannot delete {definition.token} spec: {name}",
                    "Dependent specs:",
                    *dependents,
                ]
            )
        )
    path.unlink()
    return dependents


def authorable_group_definition(group_token: str) -> ConfigGroupDefinition:
    try:
        definition = _GROUP_BY_TOKEN[group_token]
    except KeyError as exc:
        raise ValueError(f"Unsupported config group: {group_token}") from exc
    if not definition.authorable:
        raise ValueError(f"Config group is not authorable in phase 2a: {group_token}")
    return definition


def validated_model_from_disk(
    *,
    name: str,
    definition: ConfigGroupDefinition,
) -> ConfigModel:
    path = spec_path(definition.directory, name)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {definition.token} spec: {name}")
    return validate_mapping_for_write(
        definition=definition,
        name=name,
        payload=load_yaml_mapping(path),
    )


def validate_mapping_for_write(
    *,
    definition: ConfigGroupDefinition,
    name: str,
    payload: dict[str, object],
) -> ConfigModel:
    model = (
        coerce_problem_spec(payload)
        if definition.model_type is ProblemSpec
        else (
            coerce_feature_set_config(payload)
            if definition.model_type is FeatureSetConfig
            else (
                coerce_prediction_config(payload)
                if definition.model_type is PredictionConfig
                else definition.model_type.model_validate(payload)
            )
        )
    )
    _validate_identity(definition=definition, name=name, model=model)
    _validate_cross_references(definition=definition, model=model)
    return model


def write_named_group(
    *,
    definition: ConfigGroupDefinition,
    name: str,
    model: ConfigModel,
) -> Path:
    path = spec_path(definition.directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_canonical_yaml(model), encoding="utf-8")
    return path


def dump_canonical_yaml(model: ConfigModel) -> str:
    payload = _canonicalize_model(model)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def dependent_specs(*, group_token: str, name: str) -> list[str]:
    if group_token == ConfigGroup.CHAIN.value:
        dependents: list[str] = []
        for provider_name in list_group_names("provider"):
            provider = ProviderSpec.model_validate(load_named_group(provider_name, "provider"))
            if name in provider.chains:
                dependents.append(f"provider: {provider_name}")
        for preset_name in list_group_names("preset"):
            preset = PresetSpec.model_validate(load_named_group(preset_name, "preset"))
            if preset.chain == name:
                dependents.append(f"preset: {preset_name}")
        return dependents
    if group_token in {
        ConfigGroup.PROVIDER.value,
        ConfigGroup.DATASET.value,
        ConfigGroup.PROBLEM.value,
        ConfigGroup.EXECUTION.value,
        ConfigGroup.FEATURE_SET.value,
    }:
        preset_field = {
            ConfigGroup.PROVIDER.value: "provider",
            ConfigGroup.DATASET.value: "dataset",
            ConfigGroup.PROBLEM.value: "problem",
            ConfigGroup.EXECUTION.value: "execution",
            ConfigGroup.FEATURE_SET.value: "feature_set",
        }[group_token]
        dependents = []
        for preset_name in list_group_names("preset"):
            preset = PresetSpec.model_validate(load_named_group(preset_name, "preset"))
            if getattr(preset, preset_field) == name:
                dependents.append(f"preset: {preset_name}")
        return dependents
    return []


def _seed_payload(definition: ConfigGroupDefinition, name: str) -> dict[str, object]:
    if definition.identity_field is None:
        return {}
    return {definition.identity_field: name}


def _apply_set_operations(payload: dict[str, object], set_values: list[str]) -> None:
    for assignment in set_values:
        path, value = _parse_assignment(assignment)
        _set_path(payload, path.split("."), value)


def _apply_unset_operations(payload: dict[str, object], unset_paths: list[str]) -> None:
    for unset_path in unset_paths:
        segments = _path_segments(unset_path)
        _unset_path(payload, segments)


def _parse_assignment(assignment: str) -> tuple[str, object]:
    path, separator, raw_value = assignment.partition("=")
    if separator == "" or not path:
        raise ValueError(f"Invalid assignment: {assignment}")
    return ".".join(_path_segments(path)), yaml.safe_load(raw_value)


def _path_segments(path: str) -> list[str]:
    segments = [segment for segment in path.split(".") if segment]
    if not segments:
        raise ValueError(f"Invalid path: {path}")
    return segments


def _set_path(payload: dict[str, object], segments: list[str], value: object) -> None:
    current = payload
    for segment in segments[:-1]:
        existing = current.get(segment)
        if existing is None:
            nested: dict[str, object] = {}
            current[segment] = nested
            current = nested
            continue
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested path through non-mapping segment: {segment}")
        current = existing
    current[segments[-1]] = value


def _unset_path(payload: dict[str, object], segments: list[str]) -> None:
    current = payload
    for segment in segments[:-1]:
        existing = current.get(segment)
        if not isinstance(existing, dict):
            raise ValueError(f"Missing unset path: {'.'.join(segments)}")
        current = existing
    leaf = segments[-1]
    if leaf not in current:
        raise ValueError(f"Missing unset path: {'.'.join(segments)}")
    del current[leaf]


def _validate_identity(
    *,
    definition: ConfigGroupDefinition,
    name: str,
    model: ConfigModel,
) -> None:
    if definition.identity_field is None:
        return
    value = getattr(model, definition.identity_field)
    if value != name:
        raise ValueError(
            f"{definition.token} {definition.identity_field} must match spec name: {name}"
        )


def _validate_cross_references(
    *,
    definition: ConfigGroupDefinition,
    model: ConfigModel,
) -> None:
    if definition.directory == "provider":
        provider = model
        assert isinstance(provider, ProviderSpec)
        known_chains = set(list_group_names("chain"))
        unknown_chains = sorted(set(provider.chains) - known_chains)
        if unknown_chains:
            raise ValueError(
                f"provider {provider.name} declares unknown chains: {', '.join(unknown_chains)}"
            )
        return
    if definition.directory == "preset":
        preset = model
        assert isinstance(preset, PresetSpec)
        for field_name, group_name in _PRESET_REFERENCE_GROUPS.items():
            value = getattr(preset, field_name)
            if value is None:
                continue
            if value not in set(list_group_names(group_name)):
                raise ValueError(
                    f"preset.{field_name} references unknown {group_name} spec: {value}"
                )


def _canonicalize_model(model: BaseModel) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field_name in type(model).model_fields:
        value = getattr(model, field_name)
        if value is None:
            continue
        payload[field_name] = _canonicalize_value(value)
    return payload


def _canonicalize_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonicalize_model(value)
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if child is not None
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value

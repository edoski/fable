# pyright: strict

"""Config query and file-edit helpers for the fixed YAML spec set."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ValidationError

from ..core.errors import ConfigResolutionError
from ..modeling.families.registry import coerce_model_config
from .models import (
    ChainSpec,
    DatasetSpec,
    ExecutionSpec,
    PresetSpec,
    ProviderSpec,
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)

_PACKAGE_CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
_CONF_ROOT = _PACKAGE_CONF_ROOT


class ConfigGroup(StrEnum):
    CHAIN = "chain"
    DATASET = "dataset"
    DATASET_BUILDER = "dataset-builder"
    EXECUTION = "execution"
    FEATURE_SET = "feature-set"
    MODEL = "model"
    PREDICTION = "prediction"
    PRESET = "preset"
    PROBLEM = "problem"
    PROVIDER = "provider"
    TUNING_SPACE = "tuning-space"


ValidateGroupPayload = Callable[[dict[str, object]], BaseModel | dict[str, object]]


@dataclass(frozen=True, slots=True)
class GroupSpec:
    token: str
    directory: str
    seed_name: str
    validate: ValidateGroupPayload
    identity_field: str | None = None
    seed_from_requested_name: bool = False


_GROUP_SPECS = (
    GroupSpec(
        token=ConfigGroup.CHAIN.value,
        directory="chain",
        seed_name="ethereum",
        validate=ChainSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.DATASET.value,
        directory="dataset",
        seed_name="icdcs_2026",
        validate=DatasetSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.DATASET_BUILDER.value,
        directory="dataset_builder",
        seed_name="standard_temporal",
        validate=coerce_dataset_builder_config,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.EXECUTION.value,
        directory="execution",
        seed_name="disi_l40",
        validate=ExecutionSpec.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.FEATURE_SET.value,
        directory="feature_set",
        seed_name="icdcs_2026",
        validate=coerce_feature_set_config,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.MODEL.value,
        directory="model",
        seed_name="lstm",
        validate=coerce_model_config,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.PREDICTION.value,
        directory="prediction",
        seed_name="candidate_offset_selection",
        validate=coerce_prediction_config,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.PRESET.value,
        directory="preset",
        seed_name="icdcs_2026",
        validate=PresetSpec.model_validate,
    ),
    GroupSpec(
        token=ConfigGroup.PROBLEM.value,
        directory="problem",
        seed_name="icdcs_2026",
        validate=coerce_problem_spec,
        identity_field="id",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.PROVIDER.value,
        directory="provider",
        seed_name="publicnode",
        validate=ProviderSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
    ),
    GroupSpec(
        token=ConfigGroup.TUNING_SPACE.value,
        directory="tuning_space",
        seed_name="lstm_default",
        validate=lambda payload: _canonicalize_mapping(payload),
        seed_from_requested_name=True,
    ),
)
_GROUP_SPEC_BY_TOKEN = {spec.token: spec for spec in _GROUP_SPECS}
_GROUP_SPEC_BY_DIRECTORY = {spec.directory: spec for spec in _GROUP_SPECS}
_NAMED_GROUP_KEYS = tuple(
    spec.directory for spec in _GROUP_SPECS if spec.directory != "preset"
)


def config_root() -> Path:
    return _CONF_ROOT


def named_group_keys() -> tuple[str, ...]:
    return _NAMED_GROUP_KEYS


def _group_spec(group: str) -> GroupSpec:
    if group in _GROUP_SPEC_BY_TOKEN:
        return _GROUP_SPEC_BY_TOKEN[group]
    if group in _GROUP_SPEC_BY_DIRECTORY:
        return _GROUP_SPEC_BY_DIRECTORY[group]
    raise ConfigResolutionError(f"Unsupported config group: {group}")


def normalize_group_name(group: str) -> str:
    return _group_spec(group).directory


def group_path(group: str) -> Path:
    return config_root() / normalize_group_name(group)


def spec_path(group: str, name: str) -> Path:
    return group_path(group) / f"{name}.yaml"


def load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigResolutionError(f"Invalid YAML: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError(f"Configuration must be a mapping: {path}")
    return _mapping_payload(cast(Mapping[object, object], payload))


def load_named_group(name: str, group: str) -> dict[str, object]:
    path = spec_path(group, name)
    if not path.is_file():
        raise ConfigResolutionError(f"Unknown {normalize_group_name(group)} spec: {name}")
    return load_yaml_mapping(path)


def list_group_names(group: str) -> list[str]:
    return sorted(path.stem for path in group_path(group).glob("*.yaml"))


def show_named_group(group: str, name: str) -> str:
    normalized_group = normalize_group_name(group)
    payload = load_named_group(name, normalized_group)
    validated = _validate_payload(normalized_group, name=name, payload=payload)
    return dump_canonical_yaml(validated)


def ensure_named_group_file(group: str, name: str) -> Path:
    normalized_group = normalize_group_name(group)
    path = spec_path(normalized_group, name)
    if path.is_file():
        return path
    payload = _seed_payload(normalized_group, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_canonical_yaml(payload), encoding="utf-8")
    return path


def dump_canonical_yaml(value: BaseModel | dict[str, object]) -> str:
    payload = _canonicalize_value(value)
    if not isinstance(payload, dict):
        raise ConfigResolutionError("Canonical YAML root must be a mapping")
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _seed_payload(group: str, *, name: str) -> dict[str, object]:
    spec = _group_spec(group)
    template = _seed_template(group, name=name)
    identity_field = spec.identity_field
    if identity_field is not None:
        template[identity_field] = name
    validated = _validate_payload(group, name=name, payload=template)
    if isinstance(validated, BaseModel):
        return _canonicalize_model(validated)
    return _canonicalize_mapping(validated)


def _seed_template(group: str, *, name: str) -> dict[str, object]:
    spec = _group_spec(group)
    package_group_root = _PACKAGE_CONF_ROOT / spec.directory
    candidates: list[Path] = []
    if spec.seed_from_requested_name:
        candidates.append(package_group_root / f"{name}.yaml")
    candidates.append(package_group_root / f"{spec.seed_name}.yaml")
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return load_yaml_mapping(candidate)
    raise ConfigResolutionError(f"Missing seed template for config group: {group}")


def _validate_payload(
    group: str,
    *,
    name: str,
    payload: dict[str, object],
) -> BaseModel | dict[str, object]:
    spec = _group_spec(group)
    try:
        validated = spec.validate(payload)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
    if spec.identity_field is None:
        return validated
    if isinstance(validated, BaseModel):
        identity_value = getattr(validated, spec.identity_field)
    else:
        identity_value = validated.get(spec.identity_field)
    if identity_value != name:
        raise ConfigResolutionError(f"{group} {spec.identity_field} must match spec name: {name}")
    return validated


def _canonicalize_model(model: BaseModel) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field_name in type(model).model_fields:
        value = getattr(model, field_name)
        if value is None:
            continue
        payload[field_name] = _canonicalize_value(value)
    return payload


def _canonicalize_mapping(payload: dict[str, object]) -> dict[str, object]:
    return {
        str(key): _canonicalize_value(value)
        for key, value in payload.items()
        if value is not None
    }


def _canonicalize_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonicalize_model(value)
    if isinstance(value, Mapping):
        return _canonicalize_unknown_mapping(cast(Mapping[object, object], value))
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item) for item in cast(Sequence[object], value)]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _mapping_payload(payload: Mapping[object, object]) -> dict[str, object]:
    return {str(key): value for key, value in payload.items()}


def _canonicalize_unknown_mapping(payload: Mapping[object, object]) -> dict[str, object]:
    return {
        str(key): _canonicalize_value(value)
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
        if value is not None
    }

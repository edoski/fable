# pyright: strict

"""Raw config group loading and file-edit helpers."""

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
from ..evaluation import coerce_evaluator_config
from ..execution.models import ExecutionSpec
from ..modeling.dataset_builders import (
    coerce_dataset_builder_config,
)
from ..modeling.families.registry import coerce_model_config
from ..objectives import coerce_objective_config
from .models import (
    ChainSpec,
    DatasetSpec,
    PredictionConfig,
    ProviderSpec,
    SplitConfig,
    TrainingConfig,
    TuningConfig,
    coerce_features_config,
    coerce_problem_spec,
)

_PACKAGE_CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
_CONF_ROOT = _PACKAGE_CONF_ROOT


class ConfigGroup(StrEnum):
    BENCHMARK = "benchmark"
    CHAIN = "chain"
    DATASET = "dataset"
    DATASET_BUILDER = "dataset-builder"
    EVALUATION = "evaluation"
    EXECUTION = "execution"
    FEATURES = "features"
    MODEL = "model"
    OBJECTIVE = "objective"
    PREDICTION = "prediction"
    PROBLEM = "problem"
    PROVIDER = "provider"
    SPLIT = "split"
    SURFACE = "surface"
    TRAINING = "training"
    TUNING = "tuning"
    TUNING_SPACE = "tuning-space"


_ValidateGroupPayload = Callable[[dict[str, object]], BaseModel | dict[str, object]]


@dataclass(frozen=True, slots=True)
class _GroupSpec:
    token: str
    directory: str
    seed_name: str | None
    validate: _ValidateGroupPayload
    identity_field: str | None = None
    seed_from_requested_name: bool = False
    public: bool = False


def _validate_surface_frame(payload: dict[str, object]) -> BaseModel:
    from .surfaces import SurfaceFrame

    return SurfaceFrame.model_validate(payload)


_GROUP_SPECS = (
    _GroupSpec(
        token=ConfigGroup.SURFACE.value,
        directory="surface",
        seed_name="current_row_fee_dynamics",
        validate=_validate_surface_frame,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.BENCHMARK.value,
        directory="benchmark",
        seed_name=None,
        validate=lambda payload: _canonicalize_mapping(payload),
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.TRAINING.value,
        directory="training",
        seed_name="default",
        validate=TrainingConfig.model_validate,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.SPLIT.value,
        directory="split",
        seed_name="default",
        validate=SplitConfig.model_validate,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.TUNING.value,
        directory="tuning",
        seed_name="default",
        validate=TuningConfig.model_validate,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.DATASET.value,
        directory="dataset",
        seed_name="icdcs_2026",
        validate=DatasetSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.CHAIN.value,
        directory="chain",
        seed_name="ethereum",
        validate=ChainSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.PROBLEM.value,
        directory="problem",
        seed_name="current_row_nominal",
        validate=coerce_problem_spec,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.PROVIDER.value,
        directory="provider",
        seed_name="publicnode",
        validate=ProviderSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.DATASET_BUILDER.value,
        directory="dataset_builder",
        seed_name="fixed_sequence_temporal",
        validate=coerce_dataset_builder_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.EVALUATION.value,
        directory="evaluation",
        seed_name="poisson_replay_2h",
        validate=coerce_evaluator_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.EXECUTION.value,
        directory="execution",
        seed_name="disi_l40",
        validate=ExecutionSpec.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.FEATURES.value,
        directory="features",
        seed_name="core_fee_dynamics",
        validate=coerce_features_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.MODEL.value,
        directory="model",
        seed_name="lstm",
        validate=coerce_model_config,
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.OBJECTIVE.value,
        directory="objective",
        seed_name="validation_total_loss",
        validate=coerce_objective_config,
        seed_from_requested_name=False,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.PREDICTION.value,
        directory="prediction",
        seed_name="icdcs_2026",
        validate=PredictionConfig.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    _GroupSpec(
        token=ConfigGroup.TUNING_SPACE.value,
        directory="tuning_space",
        seed_name="lstm_default",
        validate=lambda payload: _canonicalize_mapping(payload),
        seed_from_requested_name=True,
        public=True,
    ),
)
_GROUP_SPEC_BY_TOKEN = {spec.token: spec for spec in _GROUP_SPECS}
_GROUP_SPEC_BY_DIRECTORY = {spec.directory: spec for spec in _GROUP_SPECS}
_NAMED_GROUP_KEYS = tuple(
    spec.directory for spec in _GROUP_SPECS if spec.directory != "benchmark"
)
_PUBLIC_GROUP_TOKENS = tuple(spec.token for spec in _GROUP_SPECS if spec.public)
_PUBLIC_GROUP_DIRECTORIES = tuple(
    _GROUP_SPEC_BY_TOKEN[token].directory for token in _PUBLIC_GROUP_TOKENS
)


def config_root() -> Path:
    return _CONF_ROOT


def named_group_keys() -> tuple[str, ...]:
    return _NAMED_GROUP_KEYS


def public_group_tokens() -> tuple[str, ...]:
    return _PUBLIC_GROUP_TOKENS


def public_group_help() -> str:
    return "One of: " + ", ".join(public_group_tokens()) + "."


def _group_spec(group: str) -> _GroupSpec:
    if group in _GROUP_SPEC_BY_TOKEN:
        return _GROUP_SPEC_BY_TOKEN[group]
    if group in _GROUP_SPEC_BY_DIRECTORY:
        return _GROUP_SPEC_BY_DIRECTORY[group]
    raise ConfigResolutionError(f"Unsupported config group: {group}")


def normalize_group_name(group: str) -> str:
    return _group_spec(group).directory


def normalize_public_group_name(group: str) -> str:
    spec = _group_spec(group)
    if spec.directory not in _PUBLIC_GROUP_DIRECTORIES:
        raise ConfigResolutionError(f"Config group is internal-only: {spec.token}")
    return spec.directory


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


def _load_named_group_validated(name: str, group: str) -> BaseModel | dict[str, object]:
    normalized_group = normalize_group_name(group)
    path = spec_path(normalized_group, name)
    if not path.is_file():
        raise ConfigResolutionError(f"Unknown {normalized_group} spec: {name}")
    payload = load_yaml_mapping(path)
    return _validate_payload(normalized_group, name=name, payload=payload)


def load_named_group_payload(name: str, group: str) -> dict[str, object]:
    validated = _load_named_group_validated(name, group)
    if isinstance(validated, BaseModel):
        return _canonicalize_model(validated)
    return _canonicalize_mapping(validated)


def list_group_names(group: str) -> list[str]:
    return sorted(path.stem for path in group_path(group).glob("*.yaml"))


def show_named_group(group: str, name: str) -> str:
    return dump_canonical_yaml(load_named_group_payload(name, group))


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
    if spec.seed_name is not None:
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

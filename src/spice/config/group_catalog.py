# pyright: strict

"""Shared catalog for named config groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, ValidationError

from ..core.errors import ConfigResolutionError
from ..evaluation import coerce_evaluator_config
from ..execution.models import ExecutionSpec
from ..modeling.dataset_builders import coerce_dataset_builder_config
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

ConfigT = TypeVar("ConfigT")
_ValidateGroupPayload = Callable[[dict[str, object]], ConfigT]


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


@dataclass(frozen=True, slots=True)
class GroupSpec(Generic[ConfigT]):
    group: ConfigGroup
    directory: str
    seed_name: str | None
    validate: _ValidateGroupPayload[ConfigT]
    identity_field: str | None = None
    seed_from_requested_name: bool = False
    public: bool = False

    @property
    def token(self) -> str:
        return self.group.value


def _validate_surface_frame(payload: dict[str, object]) -> BaseModel:
    from .surfaces import SurfaceFrame

    return SurfaceFrame.model_validate(payload)


def _mapping_payload(payload: dict[str, object]) -> dict[str, object]:
    return payload


GROUP_SPECS: tuple[GroupSpec[object], ...] = (
    GroupSpec(
        group=ConfigGroup.SURFACE,
        directory="surface",
        seed_name="current_row_fee_dynamics",
        validate=_validate_surface_frame,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.BENCHMARK,
        directory="benchmark",
        seed_name=None,
        validate=_mapping_payload,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.TRAINING,
        directory="training",
        seed_name="default",
        validate=TrainingConfig.model_validate,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.SPLIT,
        directory="split",
        seed_name="default",
        validate=SplitConfig.model_validate,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.TUNING,
        directory="tuning",
        seed_name="default",
        validate=TuningConfig.model_validate,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.DATASET,
        directory="dataset",
        seed_name="icdcs_2026",
        validate=DatasetSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.CHAIN,
        directory="chain",
        seed_name="ethereum",
        validate=ChainSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.PROBLEM,
        directory="problem",
        seed_name="current_row_nominal",
        validate=coerce_problem_spec,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.PROVIDER,
        directory="provider",
        seed_name="publicnode",
        validate=ProviderSpec.model_validate,
        identity_field="name",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.DATASET_BUILDER,
        directory="dataset_builder",
        seed_name="fixed_sequence_temporal",
        validate=coerce_dataset_builder_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.EVALUATION,
        directory="evaluation",
        seed_name="poisson_replay",
        validate=coerce_evaluator_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.EXECUTION,
        directory="execution",
        seed_name="disi_l40",
        validate=ExecutionSpec.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.FEATURES,
        directory="features",
        seed_name="core_fee_dynamics",
        validate=coerce_features_config,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.MODEL,
        directory="model",
        seed_name="lstm",
        validate=coerce_model_config,
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.OBJECTIVE,
        directory="objective",
        seed_name="validation_total_loss",
        validate=coerce_objective_config,
        seed_from_requested_name=False,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.PREDICTION,
        directory="prediction",
        seed_name="icdcs_2026",
        validate=PredictionConfig.model_validate,
        identity_field="id",
        seed_from_requested_name=True,
        public=True,
    ),
    GroupSpec(
        group=ConfigGroup.TUNING_SPACE,
        directory="tuning_space",
        seed_name="lstm_default",
        validate=_mapping_payload,
        seed_from_requested_name=True,
        public=True,
    ),
)
_GROUP_SPEC_BY_TOKEN = {spec.token: spec for spec in GROUP_SPECS}
_GROUP_SPEC_BY_DIRECTORY = {spec.directory: spec for spec in GROUP_SPECS}
_NAMED_GROUP_KEYS = tuple(spec.directory for spec in GROUP_SPECS if spec.directory != "benchmark")
_PUBLIC_GROUP_TOKENS = tuple(spec.token for spec in GROUP_SPECS if spec.public)
_PUBLIC_GROUP_DIRECTORIES = tuple(
    _GROUP_SPEC_BY_TOKEN[token].directory for token in _PUBLIC_GROUP_TOKENS
)


def named_group_keys() -> tuple[str, ...]:
    return _NAMED_GROUP_KEYS


def public_group_tokens() -> tuple[str, ...]:
    return _PUBLIC_GROUP_TOKENS


def group_spec(group: str | ConfigGroup) -> GroupSpec[object]:
    group_value = group.value if isinstance(group, ConfigGroup) else group
    if group_value in _GROUP_SPEC_BY_TOKEN:
        return _GROUP_SPEC_BY_TOKEN[group_value]
    if group_value in _GROUP_SPEC_BY_DIRECTORY:
        return _GROUP_SPEC_BY_DIRECTORY[group_value]
    raise ConfigResolutionError(f"Unsupported config group: {group_value}")


def normalize_group_name(group: str | ConfigGroup) -> str:
    return group_spec(group).directory


def normalize_public_group_name(group: str | ConfigGroup) -> str:
    spec = group_spec(group)
    if spec.directory not in _PUBLIC_GROUP_DIRECTORIES:
        raise ConfigResolutionError(f"Config group is internal-only: {spec.token}")
    return spec.directory


def identity_field_for_group(group: str | ConfigGroup) -> str | None:
    return group_spec(group).identity_field


def validate_named_group_payload(
    group: str | ConfigGroup,
    *,
    name: str,
    payload: dict[str, object],
) -> BaseModel | dict[str, object]:
    spec = group_spec(group)
    try:
        validated = spec.validate(payload)
    except ConfigResolutionError:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigResolutionError(str(exc)) from exc
    if isinstance(validated, BaseModel):
        if spec.identity_field is None:
            return validated
        identity_value = getattr(validated, spec.identity_field)
    elif isinstance(validated, dict):
        typed_validated = cast(dict[str, object], validated)
        if spec.identity_field is None:
            return typed_validated
        identity_value = typed_validated.get(spec.identity_field)
    else:
        raise ConfigResolutionError(
            f"{spec.directory} validator must return a model or mapping"
        )
    if identity_value != name:
        raise ConfigResolutionError(
            f"{spec.directory} {spec.identity_field} must match spec name: {name}"
        )
    if isinstance(validated, BaseModel):
        return validated
    return cast(dict[str, object], validated)

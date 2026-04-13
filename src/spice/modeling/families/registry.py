"""Open registry for model-family specs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import optuna
import torch

from ...config.models import (
    TrainingPrecision,
    TunedParameterSet,
    TunedTrainingParams,
    TuningSpaceConfig,
    TuningTrainingSearchSpace,
)
from ...prediction import PredictionOutputSpec
from ..models import TemporalModel
from .base import ModelConfig, ModelTuningSpaceConfig, TunedModelParams

ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)
ModelTuningSpaceT = TypeVar("ModelTuningSpaceT", bound=ModelTuningSpaceConfig)
ModelTunedParamsT = TypeVar("ModelTunedParamsT", bound=TunedModelParams)


@dataclass(frozen=True, slots=True)
class ModelSpec(Generic[ModelConfigT, ModelTuningSpaceT, ModelTunedParamsT]):
    id: str
    input_representation: str
    family_execution_id: str
    model_config_type: type[ModelConfigT]
    tuning_space_type: type[ModelTuningSpaceT]
    tuned_params_type: type[ModelTunedParamsT]
    build_model: Callable[[int, PredictionOutputSpec, ModelConfigT], TemporalModel]
    default_precision: Callable[[torch.device], TrainingPrecision]
    auto_compile: Callable[[torch.device, str], bool]
    validate_tuning_space: Callable[[ModelConfigT, ModelTuningSpaceT], None]
    sample_model_params: Callable[[optuna.Trial, ModelTuningSpaceT], ModelTunedParamsT | None]
    apply_model_params: Callable[[ModelConfigT, ModelTunedParamsT], ModelConfigT]


_MODEL_SPECS: dict[str, ModelSpec[Any, Any, Any]] = {}
_BUILTINS_LOADED = False


def register_model_spec(spec: ModelSpec[Any, Any, Any]) -> None:
    existing = _MODEL_SPECS.get(spec.id)
    if existing is not None:
        raise ValueError(f"Duplicate model spec id: {spec.id}")
    _MODEL_SPECS[spec.id] = spec


def _ensure_builtin_model_specs_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import lstm, transformer, transformer_lstm  # noqa: F401

    _BUILTINS_LOADED = True


def model_spec(model_id: str) -> ModelSpec[Any, Any, Any]:
    _ensure_builtin_model_specs_loaded()
    try:
        return _MODEL_SPECS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_SPECS))
        raise ValueError(f"Unknown model.id: {model_id}. Known models: {known}") from exc


def coerce_model_config(payload: Mapping[str, object] | ModelConfig) -> ModelConfig:
    if isinstance(payload, ModelConfig):
        raw_payload = payload.model_dump(mode="json")
        model_id = payload.id
    else:
        raw_payload = dict(payload)
        model_id = _mapping_model_id(raw_payload)
    spec = model_spec(model_id)
    return spec.model_config_type.model_validate(raw_payload)


def coerce_tuning_space_config(
    payload: Mapping[str, object] | TuningSpaceConfig | None,
    *,
    model_config: ModelConfig,
) -> TuningSpaceConfig | None:
    if payload is None:
        return None
    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, TuningSpaceConfig) else dict(payload)
    )
    if "model" not in raw_payload:
        raise ValueError("tuning_space.model is required")
    raw_model_payload = raw_payload["model"]
    if not isinstance(raw_model_payload, Mapping):
        raise TypeError("tuning_space.model must be a mapping")
    model_id = _mapping_model_id(raw_model_payload)
    spec = model_spec(model_id)
    training_payload = raw_payload.get("training")
    training = (
        None
        if training_payload is None
        else TuningTrainingSearchSpace.model_validate(training_payload)
    )
    model = spec.tuning_space_type.model_validate(dict(raw_model_payload))
    spec.validate_tuning_space(cast(Any, model_config), cast(Any, model))
    return TuningSpaceConfig(training=training, model=model)


def coerce_tuned_parameter_set(
    payload: Mapping[str, object] | TunedParameterSet,
    *,
    model_id: str | None = None,
) -> TunedParameterSet:
    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, TunedParameterSet) else dict(payload)
    )
    training_payload = raw_payload.get("training")
    model_payload = raw_payload.get("model")
    training = (
        None if training_payload is None else TunedTrainingParams.model_validate(training_payload)
    )
    model: TunedModelParams | None = None
    if model_payload is not None:
        if not isinstance(model_payload, Mapping):
            raise TypeError("tuned model params must be a mapping")
        resolved_model_id = _mapping_model_id(model_payload)
        spec = model_spec(resolved_model_id)
        model = spec.tuned_params_type.model_validate(dict(model_payload))
        if model_id is not None and resolved_model_id != model_id:
            raise ValueError("Tuned model params id does not match model.id")
    return TunedParameterSet(training=training, model=model)


def build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: ModelConfig,
) -> TemporalModel:
    spec = model_spec(config.id)
    return spec.build_model(n_features, output_spec, cast(Any, config))


def resolve_input_representation(model_id: str) -> str:
    return model_spec(model_id).input_representation


def resolve_family_execution_id(model_id: str) -> str:
    return model_spec(model_id).family_execution_id


def resolve_default_precision(model_id: str, device: torch.device) -> TrainingPrecision:
    return model_spec(model_id).default_precision(device)


def resolve_auto_compile(model_id: str, device: torch.device, precision: str) -> bool:
    return model_spec(model_id).auto_compile(device, precision)


def sample_tuned_parameters(
    trial: optuna.Trial,
    *,
    tuning_space: TuningSpaceConfig,
) -> TunedParameterSet:
    training_params: TunedTrainingParams | None = None
    if tuning_space.training is not None:
        training_values: dict[str, float] = {}
        if tuning_space.training.learning_rate is not None:
            training_values["learning_rate"] = float(
                trial.suggest_categorical(
                    "training.learning_rate",
                    tuning_space.training.learning_rate,
                )
            )
        if tuning_space.training.weight_decay is not None:
            training_values["weight_decay"] = float(
                trial.suggest_categorical(
                    "training.weight_decay",
                    tuning_space.training.weight_decay,
                )
            )
        if training_values:
            training_params = TunedTrainingParams.model_validate(training_values)
    spec = model_spec(tuning_space.model.id)
    model_params = spec.sample_model_params(trial, cast(Any, tuning_space.model))
    return TunedParameterSet(training=training_params, model=model_params)


def apply_tuned_parameters(
    model_config: ModelConfig,
    params: TunedParameterSet,
) -> ModelConfig:
    if params.model is None:
        return model_config
    spec = model_spec(model_config.id)
    return spec.apply_model_params(cast(Any, model_config), cast(Any, params.model))


def flatten_tuned_model_params(params: TunedParameterSet) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    if params.training is not None:
        if params.training.learning_rate is not None:
            flat["training.learning_rate"] = params.training.learning_rate
        if params.training.weight_decay is not None:
            flat["training.weight_decay"] = params.training.weight_decay
    if params.model is not None:
        for key, value in params.model.model_dump(exclude={"id"}, exclude_none=True).items():
            if isinstance(value, (int, float)):
                flat[f"model.{key}"] = value
    return flat


def _mapping_model_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ValueError("model.id is required")
    return value

"""Direct model-family dispatch for the fixed in-repo model set."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import optuna
import torch

from ...core.errors import ConfigResolutionError
from ...prediction import PredictionOutputSpec
from ..models import TemporalModel
from .base import ModelConfig, ModelTuningSpaceConfig, TunedModelParams

ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)
ModelTuningSpaceT = TypeVar("ModelTuningSpaceT", bound=ModelTuningSpaceConfig)
ModelTunedParamsT = TypeVar("ModelTunedParamsT", bound=TunedModelParams)


@dataclass(frozen=True, slots=True)
class ModelSpec(Generic[ModelConfigT, ModelTuningSpaceT, ModelTunedParamsT]):
    model_config_type: type[ModelConfigT]
    tuning_space_type: type[ModelTuningSpaceT]
    tuned_params_type: type[ModelTunedParamsT]
    build_model: Callable[[int, PredictionOutputSpec, ModelConfigT], TemporalModel]
    sample_model_params: Callable[[optuna.Trial, ModelTuningSpaceT], ModelTunedParamsT | None]
    apply_model_params: Callable[[ModelConfigT, ModelTunedParamsT], ModelConfigT]
    resolve_training_precision: Callable[[torch.device], str]
    resolve_compile_enabled: Callable[[torch.device], bool]
    validate_tuning_space: Callable[[ModelConfigT, ModelTuningSpaceT], None] | None = None


def _model_spec_loaders() -> dict[str, Callable[[], ModelSpec[Any, Any, Any]]]:
    from .lstm import MODEL_SPEC as lstm_spec
    from .transformer import MODEL_SPEC as transformer_spec
    from .transformer_lstm import MODEL_SPEC as transformer_lstm_spec

    return {
        "lstm": lambda: lstm_spec,
        "transformer": lambda: transformer_spec,
        "transformer_lstm": lambda: transformer_lstm_spec,
    }


def model_spec(model_id: str) -> ModelSpec[Any, Any, Any]:
    loaders = _model_spec_loaders()
    loader = loaders.get(model_id)
    if loader is None:
        known = ", ".join(sorted(loaders))
        raise ConfigResolutionError(
            f"Unknown model.id: {model_id}. Known values: {known}"
        )
    return loader()


def coerce_model_config(payload: Mapping[str, object] | ModelConfig[str]) -> ModelConfig[str]:
    if isinstance(payload, ModelConfig):
        raw_payload = payload.model_dump(mode="json")
        model_id = payload.id
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
        model_id = _mapping_id(raw_payload, field_name="model.id", label="model")
    else:
        raise ConfigResolutionError("model must be a mapping")
    return model_spec(model_id).model_config_type.model_validate(raw_payload)


def build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: ModelConfig[str],
) -> TemporalModel:
    spec = model_spec(config.id)
    return spec.build_model(n_features, output_spec, cast(Any, config))


def resolve_model_training_precision(
    *,
    device: torch.device,
    model_config: ModelConfig[str],
) -> str:
    return model_spec(model_config.id).resolve_training_precision(device)


def resolve_model_compile_enabled(
    *,
    device: torch.device,
    model_config: ModelConfig[str],
) -> bool:
    return model_spec(model_config.id).resolve_compile_enabled(device)


def apply_model_tuned_parameters(
    model_config: ModelConfig[str],
    params: TunedModelParams,
) -> ModelConfig[str]:
    spec = model_spec(model_config.id)
    return spec.apply_model_params(cast(Any, model_config), cast(Any, params))


def _mapping_id(payload: Mapping[str, object], *, field_name: str, label: str) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError(f"{field_name} is required")
    if not value:
        raise ConfigResolutionError(f"{label}.id must be a non-empty string")
    return value

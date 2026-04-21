"""Transformer-LSTM model-family spec."""

from __future__ import annotations

from typing import Literal

import optuna
from pydantic import Field, field_validator, model_validator

from ...prediction import PredictionOutputSpec
from .._runtime import require_cuda_device
from ..models import TemporalModel, TransformerLSTMBaseline
from .base import ModelConfig, ModelTuningSpaceConfig, TunedModelParams
from .registry import ModelSpec


class TransformerLstmModelConfig(ModelConfig[Literal["transformer_lstm"]]):
    id: Literal["transformer_lstm"] = "transformer_lstm"
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    d_model: int = Field(gt=0)
    nhead: int = Field(gt=0)
    transformer_layers: int = Field(gt=0)
    feedforward_dim: int = Field(gt=0)
    head_hidden_dim: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_transformer_dimensions(self) -> TransformerLstmModelConfig:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encodings")
        return self


class TransformerLstmTuningSpaceModelConfig(ModelTuningSpaceConfig[Literal["transformer_lstm"]]):
    id: Literal["transformer_lstm"] = "transformer_lstm"
    hidden_size: list[int] | None = Field(default=None, min_length=1)
    d_model: list[int] | None = Field(default=None, min_length=1)
    dropout: list[float] | None = Field(default=None, min_length=1)

    @field_validator("hidden_size", "d_model")
    @classmethod
    def validate_int_candidates(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("tuning_space.model integer candidates must be positive")
        return values

    @field_validator("dropout")
    @classmethod
    def validate_dropout_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("tuning_space.model.dropout values must be in [0.0, 1.0)")
        return values


class TransformerLstmTunedModelParams(TunedModelParams[Literal["transformer_lstm"]]):
    id: Literal["transformer_lstm"] = "transformer_lstm"
    hidden_size: int | None = Field(default=None, gt=0)
    d_model: int | None = Field(default=None, gt=0)
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> TransformerLstmTunedModelParams:
        if self.hidden_size is None and self.d_model is None and self.dropout is None:
            raise ValueError("tuned model params must declare at least one field")
        return self


def _build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: TransformerLstmModelConfig,
) -> TemporalModel:
    return TransformerLSTMBaseline(n_features, output_spec, config)


def _validate_tuning_space(
    model_config: TransformerLstmModelConfig,
    tuning_space: TransformerLstmTuningSpaceModelConfig,
) -> None:
    if tuning_space.d_model is not None:
        for value in tuning_space.d_model:
            if value % model_config.nhead != 0:
                raise ValueError("tuning_space.model.d_model values must be divisible by nhead")
            if value % 2 != 0:
                raise ValueError("tuning_space.model.d_model values must be even")


def _sample_model_params(
    trial: optuna.Trial,
    tuning_space: TransformerLstmTuningSpaceModelConfig,
) -> TransformerLstmTunedModelParams | None:
    values: dict[str, float | int] = {}
    if tuning_space.hidden_size is not None:
        values["hidden_size"] = int(
            trial.suggest_categorical("model.hidden_size", tuning_space.hidden_size)
        )
    if tuning_space.d_model is not None:
        values["d_model"] = int(trial.suggest_categorical("model.d_model", tuning_space.d_model))
    if tuning_space.dropout is not None:
        values["dropout"] = float(trial.suggest_categorical("model.dropout", tuning_space.dropout))
    if not values:
        return None
    return TransformerLstmTunedModelParams.model_validate({"id": "transformer_lstm", **values})


def _apply_model_params(
    model_config: TransformerLstmModelConfig,
    params: TransformerLstmTunedModelParams,
) -> TransformerLstmModelConfig:
    updates = params.model_dump(exclude={"id"}, exclude_none=True)
    return model_config.model_copy(update=updates)


def _resolve_training_precision(device) -> str:
    require_cuda_device(device)
    return "32-true"


def _resolve_compile_enabled(device) -> bool:
    require_cuda_device(device)
    return False


MODEL_SPEC = ModelSpec(
    model_config_type=TransformerLstmModelConfig,
    tuning_space_type=TransformerLstmTuningSpaceModelConfig,
    tuned_params_type=TransformerLstmTunedModelParams,
    build_model=_build_model,
    validate_tuning_space=_validate_tuning_space,
    sample_model_params=_sample_model_params,
    apply_model_params=_apply_model_params,
    resolve_training_precision=_resolve_training_precision,
    resolve_compile_enabled=_resolve_compile_enabled,
)

"""LSTM model-family spec."""

from __future__ import annotations

from typing import Literal

import optuna
from pydantic import Field, field_validator, model_validator

from ...prediction import PredictionOutputSpec
from .._runtime import require_cuda_device
from ..models import LSTMBaseline, TemporalModel
from .base import ModelConfig, ModelTuningSpaceConfig, TunedModelParams
from .registry import ModelSpec


class LstmModelConfig(ModelConfig[Literal["lstm"]]):
    id: Literal["lstm"] = "lstm"
    input_projection_dim: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    head_hidden_dim: int = Field(gt=0)


class LstmTuningSpaceModelConfig(ModelTuningSpaceConfig[Literal["lstm"]]):
    id: Literal["lstm"] = "lstm"
    input_projection_dim: list[int] | None = Field(default=None, min_length=1)
    hidden_size: list[int] | None = Field(default=None, min_length=1)
    num_layers: list[int] | None = Field(default=None, min_length=1)
    head_hidden_dim: list[int] | None = Field(default=None, min_length=1)
    dropout: list[float] | None = Field(default=None, min_length=1)

    @field_validator("input_projection_dim", "hidden_size", "num_layers", "head_hidden_dim")
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


class LstmTunedModelParams(TunedModelParams[Literal["lstm"]]):
    id: Literal["lstm"] = "lstm"
    input_projection_dim: int | None = Field(default=None, gt=0)
    hidden_size: int | None = Field(default=None, gt=0)
    num_layers: int | None = Field(default=None, gt=0)
    head_hidden_dim: int | None = Field(default=None, gt=0)
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> LstmTunedModelParams:
        if (
            self.input_projection_dim is None
            and self.hidden_size is None
            and self.num_layers is None
            and self.head_hidden_dim is None
            and self.dropout is None
        ):
            raise ValueError("tuned model params must declare at least one field")
        return self


def _build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: LstmModelConfig,
) -> TemporalModel:
    return LSTMBaseline(n_features, output_spec, config)


def _sample_model_params(
    trial: optuna.Trial,
    tuning_space: LstmTuningSpaceModelConfig,
) -> LstmTunedModelParams | None:
    values: dict[str, float | int] = {}
    if tuning_space.input_projection_dim is not None:
        values["input_projection_dim"] = int(
            trial.suggest_categorical(
                "model.input_projection_dim",
                tuning_space.input_projection_dim,
            )
        )
    if tuning_space.hidden_size is not None:
        values["hidden_size"] = int(
            trial.suggest_categorical("model.hidden_size", tuning_space.hidden_size)
        )
    if tuning_space.num_layers is not None:
        values["num_layers"] = int(
            trial.suggest_categorical("model.num_layers", tuning_space.num_layers)
        )
    if tuning_space.head_hidden_dim is not None:
        values["head_hidden_dim"] = int(
            trial.suggest_categorical("model.head_hidden_dim", tuning_space.head_hidden_dim)
        )
    if tuning_space.dropout is not None:
        values["dropout"] = float(trial.suggest_categorical("model.dropout", tuning_space.dropout))
    if not values:
        return None
    return LstmTunedModelParams.model_validate({"id": "lstm", **values})


def _apply_model_params(
    model_config: LstmModelConfig,
    params: LstmTunedModelParams,
) -> LstmModelConfig:
    updates = params.model_dump(exclude={"id"}, exclude_none=True)
    return model_config.model_copy(update=updates)


def _resolve_training_precision(device) -> str:
    require_cuda_device(device)
    return "32-true"


def _resolve_compile_enabled(device) -> bool:
    require_cuda_device(device)
    return False


MODEL_SPEC = ModelSpec(
    model_config_type=LstmModelConfig,
    tuning_space_type=LstmTuningSpaceModelConfig,
    tuned_params_type=LstmTunedModelParams,
    build_model=_build_model,
    sample_model_params=_sample_model_params,
    apply_model_params=_apply_model_params,
    resolve_training_precision=_resolve_training_precision,
    resolve_compile_enabled=_resolve_compile_enabled,
)

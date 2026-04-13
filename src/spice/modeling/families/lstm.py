"""LSTM model-family spec."""

from __future__ import annotations

from typing import Literal

import optuna
import torch
from pydantic import Field, field_validator, model_validator

from ...config.models import (
    ModelConfig,
    ModelTuningSpaceConfig,
    TrainingPrecision,
    TunedModelParams,
)
from ..models import LSTMBaseline, TemporalModel
from ..registry import ModelSpec, register_model_spec


class LstmModelConfig(ModelConfig):
    id: Literal["lstm"] = "lstm"
    input_projection_dim: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    head_hidden_dim: int = Field(gt=0)


class LstmTuningSpaceModelConfig(ModelTuningSpaceConfig):
    id: Literal["lstm"] = "lstm"
    hidden_size: list[int] | None = Field(default=None, min_length=1)
    dropout: list[float] | None = Field(default=None, min_length=1)

    @field_validator("hidden_size")
    @classmethod
    def validate_hidden_size_candidates(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("tuning_space.model.hidden_size values must be positive")
        return values

    @field_validator("dropout")
    @classmethod
    def validate_dropout_candidates(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("tuning_space.model.dropout values must be in [0.0, 1.0)")
        return values


class LstmTunedModelParams(TunedModelParams):
    id: Literal["lstm"] = "lstm"
    hidden_size: int | None = Field(default=None, gt=0)
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_non_empty_group(self) -> LstmTunedModelParams:
        if self.hidden_size is None and self.dropout is None:
            raise ValueError("tuned model params must declare at least one field")
        return self


def _build_model(
    n_features: int,
    n_candidate_slots: int,
    config: LstmModelConfig,
) -> TemporalModel:
    return LSTMBaseline(n_features, n_candidate_slots, config)


def _default_precision(device: torch.device) -> TrainingPrecision:
    if device.type == "cpu":
        return TrainingPrecision.FP32
    if device.type == "mps":
        return TrainingPrecision.FP32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return TrainingPrecision.BF16_MIXED
    if device.type == "cuda":
        return TrainingPrecision.FP16_MIXED
    return TrainingPrecision.FP32


def _auto_compile(device: torch.device, precision: str) -> bool:
    if device.type == "mps":
        return precision == "32-true"
    return device.type in {"cpu", "cuda"}


def _validate_tuning_space(
    model_config: LstmModelConfig,
    tuning_space: LstmTuningSpaceModelConfig,
) -> None:
    del model_config, tuning_space


def _sample_model_params(
    trial: optuna.Trial,
    tuning_space: LstmTuningSpaceModelConfig,
) -> LstmTunedModelParams | None:
    values: dict[str, float | int] = {}
    if tuning_space.hidden_size is not None:
        values["hidden_size"] = int(
            trial.suggest_categorical("model.hidden_size", tuning_space.hidden_size)
        )
    if tuning_space.dropout is not None:
        values["dropout"] = float(
            trial.suggest_categorical("model.dropout", tuning_space.dropout)
        )
    if not values:
        return None
    return LstmTunedModelParams.model_validate({"id": "lstm", **values})


def _apply_model_params(
    model_config: LstmModelConfig,
    params: LstmTunedModelParams,
) -> LstmModelConfig:
    updates = params.model_dump(exclude={"id"}, exclude_none=True)
    return model_config.model_copy(update=updates)


register_model_spec(
    ModelSpec(
        id="lstm",
        input_representation="sequence_event",
        model_config_type=LstmModelConfig,
        tuning_space_type=LstmTuningSpaceModelConfig,
        tuned_params_type=LstmTunedModelParams,
        build_model=_build_model,
        default_precision=_default_precision,
        auto_compile=_auto_compile,
        validate_tuning_space=_validate_tuning_space,
        sample_model_params=_sample_model_params,
        apply_model_params=_apply_model_params,
    )
)

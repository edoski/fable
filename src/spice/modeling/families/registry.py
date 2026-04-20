"""Closed dispatch for supported model families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import optuna
import torch

from ...config.models import (
    PredictionConfig,
    ProblemSpec,
    TrainingPrecision,
    TunedParameterSet,
    TunedPredictionParams,
    TunedProblemParams,
    TunedTrainingParams,
    TuningPredictionSearchSpace,
    TuningProblemSearchSpace,
    TuningSpaceConfig,
    TuningTrainingSearchSpace,
)
from ...core.errors import ConfigResolutionError
from ...prediction import PredictionOutputSpec
from ..models import TemporalModel
from .base import ModelConfig, ModelTuningSpaceConfig, TunedModelParams

ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)
ModelTuningSpaceT = TypeVar("ModelTuningSpaceT", bound=ModelTuningSpaceConfig)
ModelTunedParamsT = TypeVar("ModelTunedParamsT", bound=TunedModelParams)


@dataclass(frozen=True, slots=True)
class ModelSpec(Generic[ModelConfigT, ModelTuningSpaceT, ModelTunedParamsT]):
    id: str
    resolve_representation_id: Callable[[ModelConfigT], str]
    model_config_type: type[ModelConfigT]
    tuning_space_type: type[ModelTuningSpaceT]
    tuned_params_type: type[ModelTunedParamsT]
    build_model: Callable[[int, PredictionOutputSpec, ModelConfigT], TemporalModel]
    default_precision: Callable[[torch.device], TrainingPrecision]
    auto_compile: Callable[[torch.device, str], bool]
    validate_tuning_space: Callable[[ModelConfigT, ModelTuningSpaceT], None]
    sample_model_params: Callable[[optuna.Trial, ModelTuningSpaceT], ModelTunedParamsT | None]
    apply_model_params: Callable[[ModelConfigT, ModelTunedParamsT], ModelConfigT]

    def representation_id_for(self, config: ModelConfigT) -> str:
        return self.resolve_representation_id(config)


_KNOWN_MODEL_IDS = ("lstm", "transformer", "transformer_lstm")


def model_spec(model_id: str) -> ModelSpec[Any, Any, Any]:
    if model_id == "lstm":
        from .lstm import MODEL_SPEC

        return MODEL_SPEC
    if model_id == "transformer":
        from .transformer import MODEL_SPEC

        return MODEL_SPEC
    if model_id == "transformer_lstm":
        from .transformer_lstm import MODEL_SPEC

        return MODEL_SPEC
    known = ", ".join(_KNOWN_MODEL_IDS)
    raise ConfigResolutionError(f"Unknown model.id: {model_id}. Known values: {known}")


def coerce_model_config(payload: Mapping[str, object] | ModelConfig[str]) -> ModelConfig[str]:
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
    model_config: ModelConfig[str],
    problem_config: ProblemSpec | None = None,
    prediction_config: PredictionConfig | None = None,
) -> TuningSpaceConfig | None:
    if payload is None:
        return None
    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, TuningSpaceConfig) else dict(payload)
    )
    if "model" not in raw_payload:
        raise ConfigResolutionError("tuning_space.model is required")
    raw_model_payload = raw_payload["model"]
    if not isinstance(raw_model_payload, Mapping):
        raise ConfigResolutionError("tuning_space.model must be a mapping")
    model_id = _mapping_model_id(raw_model_payload)
    spec = model_spec(model_id)
    training_payload = raw_payload.get("training")
    training = (
        None
        if training_payload is None
        else TuningTrainingSearchSpace.model_validate(training_payload)
    )
    problem_payload = raw_payload.get("problem")
    problem = (
        None
        if problem_payload is None
        else _coerce_problem_tuning_space(problem_payload, problem_config=problem_config)
    )
    prediction_payload = raw_payload.get("prediction")
    prediction = (
        None
        if prediction_payload is None
        else _coerce_prediction_tuning_space(
            prediction_payload,
            prediction_config=prediction_config,
        )
    )
    model = spec.tuning_space_type.model_validate(dict(raw_model_payload))
    spec.validate_tuning_space(cast(Any, model_config), cast(Any, model))
    return TuningSpaceConfig(
        training=training,
        problem=problem,
        prediction=prediction,
        model=model,
    )


def coerce_tuned_parameter_set(
    payload: Mapping[str, object] | TunedParameterSet,
    *,
    model_id: str | None = None,
) -> TunedParameterSet:
    raw_payload = (
        payload.model_dump(mode="json") if isinstance(payload, TunedParameterSet) else dict(payload)
    )
    training_payload = raw_payload.get("training")
    problem_payload = raw_payload.get("problem")
    prediction_payload = raw_payload.get("prediction")
    model_payload = raw_payload.get("model")
    training = (
        None if training_payload is None else TunedTrainingParams.model_validate(training_payload)
    )
    problem = (
        None if problem_payload is None else TunedProblemParams.model_validate(problem_payload)
    )
    prediction = (
        None
        if prediction_payload is None
        else TunedPredictionParams.model_validate(prediction_payload)
    )
    model: TunedModelParams | None = None
    if model_payload is not None:
        if not isinstance(model_payload, Mapping):
            raise ConfigResolutionError("tuned model params must be a mapping")
        resolved_model_id = _mapping_model_id(model_payload)
        spec = model_spec(resolved_model_id)
        model = spec.tuned_params_type.model_validate(dict(model_payload))
        if model_id is not None and resolved_model_id != model_id:
            raise ConfigResolutionError("Tuned model params id does not match model.id")
    return TunedParameterSet(
        training=training,
        problem=problem,
        prediction=prediction,
        model=model,
    )


def build_model(
    n_features: int,
    output_spec: PredictionOutputSpec,
    config: ModelConfig[str],
) -> TemporalModel:
    spec = model_spec(config.id)
    return spec.build_model(n_features, output_spec, cast(Any, config))


def resolve_model_representation_id(model_config: ModelConfig[str]) -> str:
    spec = model_spec(model_config.id)
    return spec.representation_id_for(cast(Any, model_config))


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
    problem_params: TunedProblemParams | None = None
    prediction_params: TunedPredictionParams | None = None
    if tuning_space.training is not None:
        training_values: dict[str, float | int] = {}
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
        if tuning_space.training.batch_size is not None:
            training_values["batch_size"] = int(
                trial.suggest_categorical(
                    "training.batch_size",
                    tuning_space.training.batch_size,
                )
            )
        if training_values:
            training_params = TunedTrainingParams.model_validate(training_values)
    if tuning_space.problem is not None:
        problem_values: dict[str, int] = {}
        if tuning_space.problem.lookback_seconds is not None:
            problem_values["lookback_seconds"] = int(
                trial.suggest_categorical(
                    "problem.lookback_seconds",
                    tuning_space.problem.lookback_seconds,
                )
            )
        if problem_values:
            problem_params = TunedProblemParams.model_validate(problem_values)
    if tuning_space.prediction is not None:
        prediction_values: dict[str, float] = {}
        if tuning_space.prediction.classification_loss_weight is not None:
            prediction_values["classification_loss_weight"] = float(
                trial.suggest_categorical(
                    "prediction.classification_loss_weight",
                    tuning_space.prediction.classification_loss_weight,
                )
            )
        if tuning_space.prediction.regression_loss_weight is not None:
            prediction_values["regression_loss_weight"] = float(
                trial.suggest_categorical(
                    "prediction.regression_loss_weight",
                    tuning_space.prediction.regression_loss_weight,
                )
            )
        if prediction_values:
            prediction_params = TunedPredictionParams.model_validate(prediction_values)
    spec = model_spec(tuning_space.model.id)
    model_params = spec.sample_model_params(trial, cast(Any, tuning_space.model))
    return TunedParameterSet(
        training=training_params,
        problem=problem_params,
        prediction=prediction_params,
        model=model_params,
    )


def apply_tuned_parameters(
    model_config: ModelConfig[str],
    params: TunedParameterSet,
) -> ModelConfig[str]:
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
        if params.training.batch_size is not None:
            flat["training.batch_size"] = params.training.batch_size
    if params.problem is not None and params.problem.lookback_seconds is not None:
        flat["problem.lookback_seconds"] = params.problem.lookback_seconds
    if params.prediction is not None:
        if params.prediction.classification_loss_weight is not None:
            flat["prediction.classification_loss_weight"] = (
                params.prediction.classification_loss_weight
            )
        if params.prediction.regression_loss_weight is not None:
            flat["prediction.regression_loss_weight"] = (
                params.prediction.regression_loss_weight
            )
    if params.model is not None:
        for key, value in params.model.model_dump(exclude={"id"}, exclude_none=True).items():
            if isinstance(value, (int, float)):
                flat[f"model.{key}"] = value
    return flat


def _coerce_problem_tuning_space(
    payload: object,
    *,
    problem_config: ProblemSpec | None,
) -> TuningProblemSearchSpace:
    if problem_config is None:
        raise ConfigResolutionError(
            "problem_config is required when tuning_space.problem is provided"
        )
    return TuningProblemSearchSpace.model_validate(payload)


def _coerce_prediction_tuning_space(
    payload: object,
    *,
    prediction_config: PredictionConfig | None,
) -> TuningPredictionSearchSpace:
    if prediction_config is None:
        raise ConfigResolutionError(
            "prediction_config is required when tuning_space.prediction is provided"
        )
    prediction = TuningPredictionSearchSpace.model_validate(payload)
    family = prediction_config.family
    unsupported_fields: list[str] = []
    if (
        prediction.classification_loss_weight is not None
        and not hasattr(family, "classification_loss_weight")
    ):
        unsupported_fields.append("classification_loss_weight")
    if (
        prediction.regression_loss_weight is not None
        and not hasattr(family, "regression_loss_weight")
    ):
        unsupported_fields.append("regression_loss_weight")
    if unsupported_fields:
        joined = ", ".join(unsupported_fields)
        raise ConfigResolutionError(
            "tuning_space.prediction fields are unsupported for "
            f"prediction.family.id={family.id}: {joined}"
        )
    return prediction


def _mapping_model_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError("model.id is required")
    return value

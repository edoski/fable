"""Typed tuning-space and tuned-parameter helpers."""

from __future__ import annotations

from collections.abc import Mapping

import optuna

from ..config.models import (
    ProblemSpec,
    TunedProblemParams,
    TunedParameterSet,
    TunedTrainingParams,
    TuningProblemSearchSpace,
    TuningSpaceConfig,
    TuningTrainingSearchSpace,
)
from ..core.errors import ConfigResolutionError
from .families.registry import model_spec


def coerce_tuning_space_config(
    payload: Mapping[str, object] | TuningSpaceConfig | None,
    *,
    model_config,
    problem_config: ProblemSpec | None = None,
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
    model_id = _mapping_id(raw_model_payload, field_name="tuning_space.model.id")
    spec = model_spec(model_id)
    training_payload = raw_payload.get("training")
    problem_payload = raw_payload.get("problem")
    training = (
        None
        if training_payload is None
        else TuningTrainingSearchSpace.model_validate(training_payload)
    )
    problem = (
        None
        if problem_payload is None
        else _coerce_problem_tuning_space(problem_payload, problem_config=problem_config)
    )
    model = spec.tuning_space_type.model_validate(dict(raw_model_payload))
    if spec.validate_tuning_space is not None:
        spec.validate_tuning_space(model_config, model)
    return TuningSpaceConfig(
        training=training,
        problem=problem,
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
    model_payload = raw_payload.get("model")
    training = (
        None if training_payload is None else TunedTrainingParams.model_validate(training_payload)
    )
    problem = (
        None if problem_payload is None else TunedProblemParams.model_validate(problem_payload)
    )
    model = None
    if model_payload is not None:
        if not isinstance(model_payload, Mapping):
            raise ConfigResolutionError("tuned model params must be a mapping")
        resolved_model_id = _mapping_id(model_payload, field_name="model.id")
        spec = model_spec(resolved_model_id)
        model = spec.tuned_params_type.model_validate(dict(model_payload))
        if model_id is not None and resolved_model_id != model_id:
            raise ConfigResolutionError("Tuned model params id does not match model.id")
    return TunedParameterSet(training=training, problem=problem, model=model)


def sample_tuned_parameters(
    trial: optuna.Trial,
    *,
    tuning_space: TuningSpaceConfig,
) -> TunedParameterSet:
    training_params: TunedTrainingParams | None = None
    problem_params: TunedProblemParams | None = None
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
    spec = model_spec(tuning_space.model.id)
    model_params = spec.sample_model_params(trial, tuning_space.model)
    return TunedParameterSet(
        training=training_params,
        problem=problem_params,
        model=model_params,
    )


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


def _mapping_id(payload: Mapping[str, object], *, field_name: str) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError(f"{field_name} is required")
    return value

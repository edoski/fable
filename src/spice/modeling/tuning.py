"""Modeling-domain helpers for tuned parameter application."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from ..config import (
    TrainConfig,
    TuneConfig,
    TunedParameterSet,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from ..storage.study import load_best_params, load_study_manifest, validate_tuned_train_request
from .families.registry import (
    apply_tuned_parameters as apply_model_tuned_parameters,
)
from .families.registry import coerce_model_config, coerce_tuning_space_config


def apply_tuned_parameters(
    config: TrainConfig | TuneConfig,
    params: TunedParameterSet,
) -> TrainConfig | TuneConfig:
    tuned_config = deepcopy(config)
    if params.training is not None:
        if params.training.learning_rate is not None:
            tuned_config.training.learning_rate = params.training.learning_rate
        if params.training.weight_decay is not None:
            tuned_config.training.weight_decay = params.training.weight_decay
    tuned_config.model = apply_model_tuned_parameters(tuned_config.model, params)
    payload = tuned_config.model_dump(mode="json")
    payload["problem"] = coerce_problem_spec(payload["problem"])
    payload["feature_set"] = coerce_feature_set_config(payload["feature_set"])
    payload["prediction"] = coerce_prediction_config(payload["prediction"])
    payload["model"] = coerce_model_config(payload["model"])
    model_type = TuneConfig if isinstance(config, TuneConfig) else TrainConfig
    if isinstance(config, TuneConfig):
        payload["tuning_space"] = coerce_tuning_space_config(
            payload["tuning_space"],
            model_config=payload["model"],
        )
    return model_type.model_validate(payload)


def apply_study_best_params(config: TrainConfig) -> TrainConfig:
    path = config.paths.study_state_db
    if path is None:
        raise ValueError("study_state_db is required for tuned artifacts")
    try:
        manifest = load_study_manifest(path)
    except ValueError as exc:
        if str(exc).startswith("Missing study manifest:"):
            raise ValueError(
                "Configured tuned study does not match the current problem, feature set, "
                "model, or study selection"
            ) from exc
        raise
    validate_tuned_train_request(config, manifest=manifest)
    try:
        params = load_best_params(path, study_name=config.study.name)
    except OSError as exc:
        raise FileNotFoundError(
            f"Best tuning params are required but missing: {path}"
        ) from exc
    return cast(TrainConfig, apply_tuned_parameters(config, params))

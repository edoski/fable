"""Modeling-domain helpers for tuned parameter application."""

from __future__ import annotations

from copy import deepcopy
from typing import overload

from ..config import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    TunedParameterSet,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from ..core.errors import ConfigResolutionError, MissingStateError
from ..storage.layout import resolve_workflow_paths
from ..storage.study_manifest import load_study_manifest, validate_tuned_train_request
from ..storage.study_optuna import load_best_params
from .families.registry import (
    apply_tuned_parameters as apply_model_tuned_parameters,
)
from .families.registry import (
    coerce_model_config,
    coerce_tuning_space_config,
)


@overload
def apply_tuned_parameters(
    config: TrainConfig,
    params: TunedParameterSet,
) -> TrainConfig: ...


@overload
def apply_tuned_parameters(
    config: TuneConfig,
    params: TunedParameterSet,
) -> TuneConfig: ...


@overload
def apply_tuned_parameters(
    config: EvaluateConfig,
    params: TunedParameterSet,
) -> EvaluateConfig: ...


def apply_tuned_parameters(
    config: TrainConfig | TuneConfig | EvaluateConfig,
    params: TunedParameterSet,
) -> TrainConfig | TuneConfig | EvaluateConfig:
    tuned_config = deepcopy(config)
    if params.training is not None:
        if params.training.learning_rate is not None:
            tuned_config.training.learning_rate = params.training.learning_rate
        if params.training.weight_decay is not None:
            tuned_config.training.weight_decay = params.training.weight_decay
        if params.training.batch_size is not None:
            tuned_config.training.batch_size = params.training.batch_size
    if params.problem is not None and params.problem.lookback_seconds is not None:
        tuned_config.problem.lookback_seconds = params.problem.lookback_seconds
    if params.prediction is not None:
        family = tuned_config.prediction.family
        if params.prediction.classification_loss_weight is not None:
            if not hasattr(family, "classification_loss_weight"):
                raise ConfigResolutionError(
                    "classification_loss_weight tuning is unsupported for "
                    f"prediction.family.id={family.id}"
                )
            family.classification_loss_weight = params.prediction.classification_loss_weight
        if params.prediction.regression_loss_weight is not None:
            if not hasattr(family, "regression_loss_weight"):
                raise ConfigResolutionError(
                    "regression_loss_weight tuning is unsupported for "
                    f"prediction.family.id={family.id}"
                )
            family.regression_loss_weight = params.prediction.regression_loss_weight
    tuned_config.model = apply_model_tuned_parameters(tuned_config.model, params)
    payload = tuned_config.model_dump(mode="json")
    payload["problem"] = coerce_problem_spec(payload["problem"])
    payload["feature_set"] = coerce_feature_set_config(payload["feature_set"])
    payload["prediction"] = coerce_prediction_config(payload["prediction"])
    resolved_model = coerce_model_config(payload["model"])
    payload["model"] = resolved_model
    if isinstance(config, TuneConfig):
        payload["tuning_space"] = coerce_tuning_space_config(
            payload["tuning_space"],
            model_config=resolved_model,
            problem_config=payload["problem"],
            prediction_config=payload["prediction"],
        )
        return TuneConfig.model_validate(payload)
    if isinstance(config, EvaluateConfig):
        return EvaluateConfig.model_validate(payload)
    return TrainConfig.model_validate(payload)


def apply_study_best_params(config: TrainConfig | EvaluateConfig) -> TrainConfig | EvaluateConfig:
    path = resolve_workflow_paths(config).study_state_db
    if path is None:
        raise ConfigResolutionError("study_state_db is required for tuned artifacts")
    try:
        manifest = load_study_manifest(path)
    except MissingStateError as exc:
        raise ConfigResolutionError(
            "Configured tuned study does not match the current problem, feature set, "
            "model, or study selection"
        ) from exc
    validate_tuned_train_request(config, manifest=manifest)
    params = load_best_params(path, study_name=config.study.name)
    return apply_tuned_parameters(config, params)

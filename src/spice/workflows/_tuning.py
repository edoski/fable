"""Typed tuning helpers and summaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config import (
    StudyConfig,
    StudyDirection,
    TrainConfig,
    TuneConfig,
    TunedParameterSet,
    TuningObjective,
)
from ..modeling.registry import (
    apply_tuned_parameters as apply_model_tuned_parameters,
)
from ..modeling.registry import (
    coerce_model_config,
    coerce_tuned_parameter_set,
    coerce_tuning_space_config,
    flatten_tuned_model_params,
)

STUDY_CONTEXT_KEY = "spice_context"
TRIAL_PARAMS_KEY = "spice_params"
TRIAL_BEST_EPOCH_KEY = "spice_best_epoch"


class TuningTrialState(StrEnum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


@dataclass(frozen=True, slots=True)
class TrialRunMetadata:
    number: int
    value: float | None
    best_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class TrialSummary:
    number: int
    value: float | None
    params: TunedParameterSet
    best_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class TrialCounts:
    total: int
    complete: int
    pruned: int
    failed: int


@dataclass(frozen=True, slots=True)
class TuningTrialRecord:
    number: int
    state: TuningTrialState
    value: float | None
    params: TunedParameterSet
    best_epoch: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TuningStudySummary:
    study: StudyConfig
    chain: str
    dataset_id: str
    task_id: str
    feature_set_id: str
    model_id: str
    max_supported_delay_seconds: int
    lookback_seconds: int
    sample_count: int
    objective_metric: TuningObjective
    direction: StudyDirection
    trial_count_requested: int
    timeout_seconds: int | None
    sampler: str
    sampler_seed: int
    pruner: str
    trial_counts: TrialCounts
    best_trial: TrialSummary | None


def study_context_payload(config: TuneConfig) -> dict[str, object]:
    return {
        "chain": config.chain.name,
        "dataset_id": config.dataset.id,
        "task_id": config.task.id,
        "feature_set_id": config.feature_set.id,
        "model_id": config.model.id,
        "max_supported_delay_seconds": config.task.max_supported_delay_seconds,
        "lookback_seconds": config.task.lookback_seconds,
        "sample_count": config.task.sample_count,
        "objective_metric": config.tuning.objective_metric.value,
        "direction": config.tuning.direction.value,
        "trial_count_requested": config.tuning.trial_count,
        "timeout_seconds": config.tuning.timeout_seconds,
        "sampler": "TPESampler",
        "sampler_seed": config.tuning.sampler_seed,
        "pruner": "MedianPruner" if config.tuning.enable_pruning else "NopPruner",
    }


def _state_from_optuna(state: TrialState) -> TuningTrialState:
    return TuningTrialState(state.name)


def _trial_params_payload(params: TunedParameterSet) -> dict[str, object]:
    payload = params.model_dump(mode="json", exclude_none=True)
    if not isinstance(payload, dict):
        raise TypeError("TunedParameterSet did not serialize to a mapping payload")
    return payload


def freeze_tuned_parameters_for_trial(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr(TRIAL_PARAMS_KEY, _trial_params_payload(params))


def freeze_best_epoch_for_trial(trial: optuna.Trial, best_epoch: int) -> None:
    trial.set_user_attr(TRIAL_BEST_EPOCH_KEY, best_epoch)


def params_from_trial(trial: FrozenTrial, *, model_id: str | None = None) -> TunedParameterSet:
    payload = trial.user_attrs.get(TRIAL_PARAMS_KEY)
    if not isinstance(payload, dict):
        raise ValueError(f"Trial {trial.number} is missing typed params metadata")
    return coerce_tuned_parameter_set(payload, model_id=model_id)


def best_epoch_from_trial(trial: FrozenTrial) -> int | None:
    value = trial.user_attrs.get(TRIAL_BEST_EPOCH_KEY)
    return value if isinstance(value, int) else None


def flatten_tuned_parameters(params: TunedParameterSet) -> dict[str, float | int]:
    return flatten_tuned_model_params(params)


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
    payload["model"] = coerce_model_config(payload["model"])
    model_type = TuneConfig if isinstance(config, TuneConfig) else TrainConfig
    if isinstance(config, TuneConfig):
        payload["tuning_space"] = coerce_tuning_space_config(
            payload["tuning_space"],
            model_config=payload["model"],
        )
    return model_type.model_validate(payload)


def build_trial_record(trial: FrozenTrial, *, model_id: str | None = None) -> TuningTrialRecord:
    return TuningTrialRecord(
        number=trial.number,
        state=_state_from_optuna(trial.state),
        value=trial.value,
        params=params_from_trial(trial, model_id=model_id),
        best_epoch=best_epoch_from_trial(trial),
        started_at=trial.datetime_start,
        completed_at=trial.datetime_complete,
    )


def _trial_summary(trial: FrozenTrial, *, model_id: str | None = None) -> TrialSummary:
    return TrialSummary(
        number=trial.number,
        value=trial.value,
        params=params_from_trial(trial, model_id=model_id),
        best_epoch=best_epoch_from_trial(trial),
    )


def build_study_summary(config: TuneConfig, study: optuna.Study) -> TuningStudySummary:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if completed_trials else None
    return TuningStudySummary(
        study=config.study,
        chain=config.chain.name,
        dataset_id=config.dataset.id,
        task_id=config.task.id,
        feature_set_id=config.feature_set.id,
        model_id=config.model.id,
        max_supported_delay_seconds=config.task.max_supported_delay_seconds,
        lookback_seconds=config.task.lookback_seconds,
        sample_count=config.task.sample_count,
        objective_metric=config.tuning.objective_metric,
        direction=config.tuning.direction,
        trial_count_requested=config.tuning.trial_count,
        timeout_seconds=config.tuning.timeout_seconds,
        sampler="TPESampler",
        sampler_seed=config.tuning.sampler_seed,
        pruner="MedianPruner" if config.tuning.enable_pruning else "NopPruner",
        trial_counts=TrialCounts(
            total=len(study.trials),
            complete=len(completed_trials),
            pruned=len(pruned_trials),
            failed=len(failed_trials),
        ),
        best_trial=(
            None
            if best_trial is None
            else _trial_summary(best_trial, model_id=config.model.id)
        ),
    )

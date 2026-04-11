"""Typed tuning helpers and artifact models."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import optuna
from optuna.trial import FrozenTrial, TrialState
from pydantic import BaseModel, ConfigDict

from ..core.config import (
    ChainName,
    ExperimentConfig,
    ModelFamily,
    StudyDirection,
    TunedModelParams,
    TunedParameterSet,
    TunedTrainingParams,
    TuningObjective,
    TuningSearchSpace,
    revalidate_config,
)


class TuningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TuningTrialState(StrEnum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


class TrialRunMetadata(TuningModel):
    number: int
    value: float | None
    best_epoch: int | None = None
    artifact_dir: Path | None = None


class TrialSummary(TuningModel):
    number: int
    value: float | None
    params: TunedParameterSet
    best_epoch: int | None = None
    artifact_dir: Path | None = None


class TrialCounts(TuningModel):
    total: int
    complete: int
    pruned: int
    failed: int


class TuningTrialRecord(TuningModel):
    number: int
    state: TuningTrialState
    value: float | None
    params: TunedParameterSet
    best_epoch: int | None = None
    artifact_dir: Path | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TuningStudyReport(TuningModel):
    kind: Literal["tuning_study"] = "tuning_study"
    study_name: str
    chain: ChainName
    dataset_id: str
    family: ModelFamily
    max_delay_seconds: int
    lookback_seconds: int
    anchor_count: int
    objective_metric: TuningObjective
    direction: StudyDirection
    trial_count_requested: int
    timeout_seconds: int | None = None
    sampler: str
    sampler_seed: int
    pruner: str
    search_space: TuningSearchSpace
    trial_counts: TrialCounts
    best_trial: TrialSummary | None


class TuningBestParamsReport(TuningModel):
    kind: Literal["tuning_best_params"] = "tuning_best_params"
    study_name: str
    chain: ChainName
    dataset_id: str
    family: ModelFamily
    max_delay_seconds: int
    lookback_seconds: int
    anchor_count: int
    objective_metric: TuningObjective
    direction: StudyDirection
    trial: TrialRunMetadata
    params: TunedParameterSet


def _state_from_optuna(state: TrialState) -> TuningTrialState:
    return TuningTrialState(state.name)


def _trial_params_payload(params: TunedParameterSet) -> dict[str, object]:
    payload = params.model_dump(mode="json", exclude_none=True)
    if not isinstance(payload, dict):
        raise TypeError("TunedParameterSet did not serialize to a mapping payload")
    return payload


def freeze_tuned_parameters_for_trial(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr("params", _trial_params_payload(params))


def _params_from_trial(trial: FrozenTrial) -> TunedParameterSet:
    payload = trial.user_attrs.get("params")
    if not isinstance(payload, dict):
        raise ValueError(f"Trial {trial.number} is missing typed params metadata")
    return TunedParameterSet.model_validate(payload)


def flatten_tuned_parameters(params: TunedParameterSet) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    if params.training is not None:
        if params.training.learning_rate is not None:
            flat["training.learning_rate"] = params.training.learning_rate
        if params.training.weight_decay is not None:
            flat["training.weight_decay"] = params.training.weight_decay
    if params.model is not None:
        if params.model.hidden_size is not None:
            flat["model.hidden_size"] = params.model.hidden_size
        if params.model.dropout is not None:
            flat["model.dropout"] = params.model.dropout
    return flat


def sample_tuned_parameters(
    trial: optuna.Trial,
    search_space: TuningSearchSpace,
) -> TunedParameterSet:
    training_params: TunedTrainingParams | None = None
    model_params: TunedModelParams | None = None

    if search_space.training is not None:
        training_values: dict[str, float] = {}
        if search_space.training.learning_rate is not None:
            training_values["learning_rate"] = float(
                trial.suggest_categorical(
                    "training.learning_rate",
                    search_space.training.learning_rate,
                )
            )
        if search_space.training.weight_decay is not None:
            training_values["weight_decay"] = float(
                trial.suggest_categorical(
                    "training.weight_decay",
                    search_space.training.weight_decay,
                )
            )
        if training_values:
            training_params = TunedTrainingParams.model_validate(training_values)

    if search_space.model is not None:
        model_values: dict[str, float | int] = {}
        if search_space.model.hidden_size is not None:
            model_values["hidden_size"] = int(
                trial.suggest_categorical(
                    "model.hidden_size",
                    search_space.model.hidden_size,
                )
            )
        if search_space.model.dropout is not None:
            model_values["dropout"] = float(
                trial.suggest_categorical(
                    "model.dropout",
                    search_space.model.dropout,
                )
            )
        if model_values:
            model_params = TunedModelParams.model_validate(model_values)

    return TunedParameterSet(training=training_params, model=model_params)


def apply_tuned_parameters(
    config: ExperimentConfig,
    params: TunedParameterSet,
) -> ExperimentConfig:
    tuned_config = deepcopy(config)
    if params.training is not None:
        if params.training.learning_rate is not None:
            tuned_config.training.learning_rate = params.training.learning_rate
        if params.training.weight_decay is not None:
            tuned_config.training.weight_decay = params.training.weight_decay
    if params.model is not None:
        if params.model.hidden_size is not None:
            tuned_config.model.hidden_size = params.model.hidden_size
        if params.model.dropout is not None:
            tuned_config.model.dropout = params.model.dropout
    return revalidate_config(tuned_config)


def build_trial_record(trial: FrozenTrial) -> TuningTrialRecord:
    params = _params_from_trial(trial)
    return TuningTrialRecord(
        number=trial.number,
        state=_state_from_optuna(trial.state),
        value=trial.value,
        params=params,
        best_epoch=_best_epoch_from_trial(trial),
        artifact_dir=_artifact_dir_from_trial(trial),
        started_at=trial.datetime_start,
        completed_at=trial.datetime_complete,
    )


def _trial_summary(trial: FrozenTrial) -> TrialSummary:
    params = _params_from_trial(trial)
    return TrialSummary(
        number=trial.number,
        value=trial.value,
        params=params,
        best_epoch=_best_epoch_from_trial(trial),
        artifact_dir=_artifact_dir_from_trial(trial),
    )


def _best_epoch_from_trial(trial: FrozenTrial) -> int | None:
    value = trial.user_attrs.get("best_epoch")
    return value if isinstance(value, int) else None


def _artifact_dir_from_trial(trial: FrozenTrial) -> Path | None:
    value = trial.user_attrs.get("artifact_dir")
    return Path(value) if isinstance(value, str) else None


def build_study_report(config: ExperimentConfig, study: optuna.Study) -> TuningStudyReport:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if completed_trials else None
    return TuningStudyReport(
        study_name=config.tuning.study_name,
        chain=config.chain.name,
        dataset_id=config.dataset.id,
        family=config.model.family,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        anchor_count=config.dataset.sampling.anchor_count,
        objective_metric=config.tuning.objective_metric,
        direction=config.tuning.direction,
        trial_count_requested=config.tuning.trial_count,
        timeout_seconds=config.tuning.timeout_seconds,
        sampler="TPESampler",
        sampler_seed=config.tuning.sampler_seed,
        pruner="MedianPruner" if config.tuning.enable_pruning else "NopPruner",
        search_space=config.tuning.search_space,
        trial_counts=TrialCounts(
            total=len(study.trials),
            complete=len(completed_trials),
            pruned=len(pruned_trials),
            failed=len(failed_trials),
        ),
        best_trial=None if best_trial is None else _trial_summary(best_trial),
    )


def build_best_params_report(
    config: ExperimentConfig,
    study: optuna.Study,
) -> TuningBestParamsReport:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Optuna study completed without any successful trials")
    best_trial = study.best_trial
    return TuningBestParamsReport(
        study_name=config.tuning.study_name,
        chain=config.chain.name,
        dataset_id=config.dataset.id,
        family=config.model.family,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        anchor_count=config.dataset.sampling.anchor_count,
        objective_metric=config.tuning.objective_metric,
        direction=config.tuning.direction,
        trial=TrialRunMetadata(
            number=best_trial.number,
            value=best_trial.value,
            best_epoch=_best_epoch_from_trial(best_trial),
            artifact_dir=_artifact_dir_from_trial(best_trial),
        ),
        params=_params_from_trial(best_trial),
    )

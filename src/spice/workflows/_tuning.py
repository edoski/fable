"""Typed tuning helpers and artifact models."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import optuna
from optuna.trial import FrozenTrial, TrialState
from pydantic import BaseModel, ConfigDict

from ..config import (
    StudyConfig,
    StudyDirection,
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
    study: StudyConfig
    chain: str
    dataset_id: str
    model_id: str
    max_delay_seconds: int
    lookback_seconds: int
    sample_count: int
    objective_metric: TuningObjective
    direction: StudyDirection
    trial_count_requested: int
    timeout_seconds: int | None = None
    sampler: str
    sampler_seed: int
    pruner: str
    trial_counts: TrialCounts
    best_trial: TrialSummary | None


class TuningBestParamsReport(TuningModel):
    kind: Literal["tuning_best_params"] = "tuning_best_params"
    study: StudyConfig
    chain: str
    dataset_id: str
    model_id: str
    max_delay_seconds: int
    lookback_seconds: int
    sample_count: int
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
    return coerce_tuned_parameter_set(payload)


def flatten_tuned_parameters(params: TunedParameterSet) -> dict[str, float | int]:
    return flatten_tuned_model_params(params)


def apply_tuned_parameters(
    config: TuneConfig,
    params: TunedParameterSet,
) -> TuneConfig:
    tuned_config = deepcopy(config)
    if params.training is not None:
        if params.training.learning_rate is not None:
            tuned_config.training.learning_rate = params.training.learning_rate
        if params.training.weight_decay is not None:
            tuned_config.training.weight_decay = params.training.weight_decay
    tuned_config.model = apply_model_tuned_parameters(tuned_config.model, params)
    payload = tuned_config.model_dump(mode="json")
    payload["model"] = coerce_model_config(payload["model"])
    payload["tuning_space"] = coerce_tuning_space_config(
        payload["tuning_space"],
        model_config=payload["model"],
    )
    return TuneConfig.model_validate(payload)


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


def build_study_report(config: TuneConfig, study: optuna.Study) -> TuningStudyReport:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if completed_trials else None
    return TuningStudyReport(
        study=config.study,
        chain=config.chain.name,
        dataset_id=config.dataset.id,
        model_id=config.model.id,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        sample_count=config.dataset.sampling.sample_count,
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
        best_trial=None if best_trial is None else _trial_summary(best_trial),
    )


def build_best_params_report(
    config: TuneConfig,
    study: optuna.Study,
) -> TuningBestParamsReport:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Optuna study completed without any successful trials")
    best_trial = study.best_trial
    return TuningBestParamsReport(
        study=config.study,
        chain=config.chain.name,
        dataset_id=config.dataset.id,
        model_id=config.model.id,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        sample_count=config.dataset.sampling.sample_count,
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


def load_tuning_best_params_report(path: Path) -> TuningBestParamsReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Tuning best params report must be a mapping")
    payload["params"] = coerce_tuned_parameter_set(
        payload["params"],
        model_id=str(payload["model_id"]),
    )
    return TuningBestParamsReport.model_validate(payload)

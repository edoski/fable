"""Study-domain models and pure summary builders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config import (
    FeatureSetConfig,
    ModelConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    TrainConfig,
    TrainingConfig,
    TunedParameterSet,
    TuningSpaceConfig,
)
from ..modeling.families.registry import coerce_tuned_parameter_set
from ..semantics import StudySemantics

TRIAL_PARAMS_KEY = "spice_params"
TRIAL_BEST_EPOCH_KEY = "spice_best_epoch"


class StudyTrialState(StrEnum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


@dataclass(frozen=True, slots=True)
class StudyManifest:
    study_id: str
    prediction: PredictionConfig
    study_name: str
    chain_name: str
    dataset_id: str
    dataset_name: str
    problem: ProblemSpec
    feature_set: FeatureSetConfig
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    sampler_name: str
    sampler_seed: int
    pruner_name: str
    enable_pruning: bool
    tuning_space: TuningSpaceConfig
    semantics: StudySemantics

    @property
    def problem_id(self) -> str:
        return self.semantics.problem.problem_id

    @property
    def feature_set_id(self) -> str:
        return self.semantics.feature.feature_set_id

    @property
    def prediction_id(self) -> str:
        return self.semantics.prediction.prediction_id

    @property
    def prediction_family_id(self) -> str:
        return self.semantics.prediction.prediction_family_id

    @property
    def model_id(self) -> str:
        return self.model.id

    @property
    def representation_id(self) -> str:
        return self.semantics.representation.representation_id

    @property
    def input_normalization_id(self) -> str:
        return self.semantics.input_normalization.input_normalization_id


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
class StudyTrialRecord:
    number: int
    state: StudyTrialState
    value: float | None
    params: TunedParameterSet
    best_epoch: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StudySummary:
    manifest: StudyManifest
    trial_counts: TrialCounts
    best_trial: TrialSummary | None


@dataclass(frozen=True, slots=True)
class OpenStudy:
    manifest: StudyManifest
    study: optuna.Study
    existing_trial_count: int
    target_trial_count: int
    remaining_trial_count: int


def params_from_trial(trial: FrozenTrial, *, model_id: str) -> TunedParameterSet:
    payload = trial.user_attrs.get(TRIAL_PARAMS_KEY)
    if not isinstance(payload, dict):
        raise ValueError(f"Trial {trial.number} is missing typed params metadata")
    return coerce_tuned_parameter_set(payload, model_id=model_id)


def best_epoch_from_trial(trial: FrozenTrial) -> int | None:
    value = trial.user_attrs.get(TRIAL_BEST_EPOCH_KEY)
    return value if isinstance(value, int) else None


def build_trial_record(trial: FrozenTrial, *, model_id: str) -> StudyTrialRecord:
    return StudyTrialRecord(
        number=trial.number,
        state=StudyTrialState(trial.state.name),
        value=trial.value,
        params=params_from_trial(trial, model_id=model_id),
        best_epoch=best_epoch_from_trial(trial),
        started_at=trial.datetime_start,
        completed_at=trial.datetime_complete,
    )


def build_study_summary(manifest: StudyManifest, study: optuna.Study) -> StudySummary:
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if completed_trials else None
    return StudySummary(
        manifest=manifest,
        trial_counts=TrialCounts(
            total=len(study.trials),
            complete=len(completed_trials),
            pruned=len(pruned_trials),
            failed=len(failed_trials),
        ),
        best_trial=(
            None
            if best_trial is None
            else TrialSummary(
                number=best_trial.number,
                value=best_trial.value,
                params=params_from_trial(best_trial, model_id=manifest.model_id),
                best_epoch=best_epoch_from_trial(best_trial),
            )
        ),
    )


def trial_params_payload(params: TunedParameterSet) -> dict[str, object]:
    payload = params.model_dump(mode="json", exclude_none=True)
    if not isinstance(payload, dict):
        raise TypeError("TunedParameterSet did not serialize to a mapping payload")
    return payload


def tuned_train_request_identity(config: TrainConfig) -> dict[str, object]:
    return {
        "study_name": config.study.name,
        "study_id": config.paths.study_id,
        "prediction": config.prediction.model_dump(mode="json"),
        "chain_name": config.chain.name,
        "dataset_id": config.paths.corpus_id,
        "dataset_name": config.dataset.name,
        "problem": config.problem.model_dump(mode="json"),
        "feature_set": config.feature_set.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json", exclude_none=True),
    }

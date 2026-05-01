"""Modeling-domain helpers for tuned parameter application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast, overload

from ..config.models import StudyConfig, TrainConfig, TuneConfig, TunedParameterSet, WorkflowTask
from ..config.workflow_snapshots import hydrate_workflow_config_snapshot
from ..core.errors import ConfigResolutionError, MissingStateError
from ..storage.root_handles import CorpusRootHandle, StudyRootHandle
from ..storage.study_manifest import load_study_manifest, validate_tuned_artifact_definition
from ..storage.study_optuna import load_best_params
from .families.registry import (
    apply_model_tuned_parameters,
)


@dataclass(frozen=True, slots=True)
class AppliedStudyBestParams:
    config: TrainConfig
    study_id: str


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
        if params.training.batch_size is not None:
            tuned_config.training.batch_size = params.training.batch_size
    if params.problem is not None and params.problem.lookback_seconds is not None:
        tuned_config.problem.lookback_seconds = params.problem.lookback_seconds
    if params.model is not None:
        tuned_config.model = apply_model_tuned_parameters(tuned_config.model, params.model)
    payload = tuned_config.model_dump(mode="json")
    if isinstance(config, TuneConfig):
        return cast(TuneConfig, hydrate_workflow_config_snapshot(WorkflowTask.TUNE, payload))
    return cast(TrainConfig, hydrate_workflow_config_snapshot(WorkflowTask.TRAIN, payload))


def apply_study_best_params(
    config: TrainConfig,
    *,
    study: StudyRootHandle,
    corpus: CorpusRootHandle,
) -> AppliedStudyBestParams:
    try:
        manifest = load_study_manifest(study.state_db_path)
    except MissingStateError as exc:
        raise ConfigResolutionError(
            "Configured tuned study does not match the current problem, features, "
            "model, or study selection"
        ) from exc
    study_config = _with_manifest_study_name(config, study_name=manifest.study_name)
    validate_tuned_artifact_definition(
        study_config,
        manifest=manifest,
        study_id=study.study_id,
        dataset_id=corpus.dataset_id,
    )
    params = load_best_params(study.state_db_path, study_name=manifest.study_name)
    tuned_config = apply_tuned_parameters(study_config, params)
    return AppliedStudyBestParams(config=tuned_config, study_id=study.study_id)


def _with_manifest_study_name(config: TrainConfig, *, study_name: str) -> TrainConfig:
    return config.model_copy(update={"study": StudyConfig(name=study_name)})

"""Study Optuna access and trial persistence."""

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.storages import RDBStorage

from ..config import TuneConfig, TunedParameterSet
from ..core.errors import MissingStateError, StateConflictError, StateLayoutError
from ..modeling.families.registry import coerce_tuned_parameter_set
from ..prediction import compile_prediction_contract
from .engine import db_url
from .study_manifest import (
    diff_study_manifests,
    insert_study_manifest,
    load_study_manifest,
    manifest_from_tune_config,
    try_load_study_manifest,
)
from .study_models import (
    TRIAL_BEST_EPOCH_KEY,
    TRIAL_PARAMS_KEY,
    OpenStudy,
    StudySummary,
    StudyTrialRecord,
    build_study_summary,
    build_trial_record,
    trial_params_payload,
)


def study_storage(db_path: Path) -> RDBStorage:
    return RDBStorage(
        url=db_url(db_path),
        engine_kwargs={"connect_args": {"timeout": 5}},
    )


def load_study(db_path: Path, *, study_name: str) -> optuna.Study:
    manifest = load_study_manifest(db_path)
    if manifest.study_name != study_name:
        raise StateConflictError(
            f"Study name mismatch: expected {manifest.study_name}, got {study_name}"
        )
    return load_materialized_study(db_path, study_name=manifest.study_name)


def open_tuning_study(db_path: Path, *, config: TuneConfig) -> OpenStudy:
    requested_manifest = manifest_from_tune_config(config)
    stored_manifest = try_load_study_manifest(db_path)
    if stored_manifest is None:
        insert_study_manifest(db_path, manifest=requested_manifest)
        manifest = requested_manifest
    else:
        mismatches = diff_study_manifests(stored_manifest, requested_manifest)
        if mismatches:
            raise StateConflictError(
                "Existing study definition does not match current request: " + ", ".join(mismatches)
            )
        manifest = stored_manifest
    study = load_or_create_materialized_study(db_path, config=config)
    existing_trial_count = len(study.trials)
    target_trial_count = config.tuning.trial_count
    if target_trial_count < existing_trial_count:
        raise StateConflictError(
            "Requested trial_count is lower than existing study size: "
            f"requested {target_trial_count}, existing {existing_trial_count}"
        )
    return OpenStudy(
        manifest=manifest,
        study=study,
        existing_trial_count=existing_trial_count,
        target_trial_count=target_trial_count,
        remaining_trial_count=target_trial_count - existing_trial_count,
    )


def load_best_params(db_path: Path, *, study_name: str) -> TunedParameterSet:
    manifest = load_study_manifest(db_path)
    if manifest.study_name != study_name:
        raise StateConflictError(
            f"Study name mismatch: expected {manifest.study_name}, got {study_name}"
        )
    study = load_materialized_study(db_path, study_name=manifest.study_name)
    payload = study.best_trial.user_attrs.get(TRIAL_PARAMS_KEY)
    if not isinstance(payload, dict):
        raise MissingStateError(f"Best tuning params are required but missing: {db_path}")
    return coerce_tuned_parameter_set(payload, model_id=manifest.model_id)


def load_study_summary(db_path: Path) -> StudySummary:
    manifest = load_study_manifest(db_path)
    study = load_materialized_study(db_path, study_name=manifest.study_name)
    return build_study_summary(manifest, study)


def list_trial_records(db_path: Path) -> list[StudyTrialRecord]:
    manifest = load_study_manifest(db_path)
    study = load_materialized_study(db_path, study_name=manifest.study_name)
    return [build_trial_record(trial, model_id=manifest.model_id) for trial in study.trials]


def record_trial_params(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr(TRIAL_PARAMS_KEY, trial_params_payload(params))


def record_trial_best_epoch(trial: optuna.Trial, best_epoch: int) -> None:
    trial.set_user_attr(TRIAL_BEST_EPOCH_KEY, best_epoch)


def load_materialized_study(db_path: Path, *, study_name: str) -> optuna.Study:
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        raise MissingStateError(f"Missing materialized Optuna study: {db_path}")
    if len(summaries) != 1:
        raise StateLayoutError(
            f"Expected exactly one Optuna study in {db_path}, found {len(summaries)}"
        )
    summary = summaries[0]
    if summary.study_name != study_name:
        raise StateLayoutError(
            "Materialized Optuna study name does not match stored manifest: "
            f"expected {study_name}, got {summary.study_name}"
        )
    return optuna.load_study(study_name=study_name, storage=study_storage(db_path))


def load_or_create_materialized_study(db_path: Path, *, config: TuneConfig) -> optuna.Study:
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        prediction_contract = compile_prediction_contract(
            prediction_id=config.prediction.id,
            family_config=config.prediction.family,
        )
        direction = (
            prediction_contract.direction
            if config.tuning.objective is None
            else config.tuning.objective.direction.value
        )
        return optuna.create_study(
            study_name=config.study.name,
            storage=study_storage(db_path),
            direction=direction,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=config.tuning.sampler_seed),
            pruner=(
                optuna.pruners.MedianPruner()
                if config.tuning.enable_pruning
                else optuna.pruners.NopPruner()
            ),
        )
    return load_materialized_study(db_path, study_name=config.study.name)

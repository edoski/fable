"""Study Optuna access and trial persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import optuna
from optuna.storages import RDBStorage

from ..config.models import TuneConfig, TunedParameterSet
from ..core.errors import MissingStateError, StateConflictError, StateLayoutError
from ..corpus.metadata import DatasetManifest
from ..modeling.tuned_config import coerce_tuned_parameter_set
from .engine import db_url
from .root_handles import CorpusRootHandle, StudyRootHandle
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
    StudyManifest,
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


def open_tuning_study(
    study: StudyRootHandle,
    *,
    config: TuneConfig,
    corpus: CorpusRootHandle,
    corpus_manifest: DatasetManifest,
) -> OpenStudy:
    requested_manifest = manifest_from_tune_config(
        config,
        corpus=corpus,
        study=study,
        corpus_manifest=corpus_manifest,
    )
    db_path = study.state_db_path
    stored_manifest = try_load_study_manifest(db_path)
    if stored_manifest is None:
        insert_study_manifest(db_path, manifest=requested_manifest)
        manifest = requested_manifest
    else:
        mismatches = diff_study_manifests(stored_manifest, requested_manifest)
        if mismatches:
            raise StateConflictError(
                "Existing study definition does not match current definition: "
                + ", ".join(mismatches)
            )
        manifest = stored_manifest
    materialized_study = load_or_create_materialized_study(db_path, config=config)
    existing_trial_count = len(materialized_study.trials)
    target_trial_count = config.tuning.trial_count
    if target_trial_count < existing_trial_count:
        raise StateConflictError(
            "Requested trial_count is lower than existing study size: "
            f"requested {target_trial_count}, existing {existing_trial_count}"
        )
    return OpenStudy(
        manifest=manifest,
        study=materialized_study,
        existing_trial_count=existing_trial_count,
        target_trial_count=target_trial_count,
        remaining_trial_count=target_trial_count - existing_trial_count,
    )


def load_best_params(db_path: Path, *, study_name: str) -> TunedParameterSet:
    manifest, study = _load_verified_study(db_path, study_name=study_name)
    payload = study.best_trial.user_attrs.get(TRIAL_PARAMS_KEY)
    if not isinstance(payload, dict):
        raise MissingStateError(f"Best tuning params are required but missing: {db_path}")
    return coerce_tuned_parameter_set(payload, model_id=manifest.model_id)


def load_study_summary(db_path: Path) -> StudySummary:
    manifest, study = _load_verified_study(db_path)
    return build_study_summary(manifest, study)


def list_trial_records(db_path: Path) -> list[StudyTrialRecord]:
    manifest, study = _load_verified_study(db_path)
    return [build_trial_record(trial, model_id=manifest.model_id) for trial in study.trials]


def record_trial_params(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr(TRIAL_PARAMS_KEY, trial_params_payload(params))


def record_trial_best_epoch(trial: optuna.Trial, best_epoch: int) -> None:
    trial.set_user_attr(TRIAL_BEST_EPOCH_KEY, best_epoch)


def load_materialized_study(db_path: Path, *, study_name: str) -> optuna.Study:
    summary = _materialized_study_summary(db_path)
    if summary is None:
        raise MissingStateError(f"Missing materialized Optuna study: {db_path}")
    if summary.study_name != study_name:
        raise StateLayoutError(
            "Materialized Optuna study name does not match stored manifest: "
            f"expected {study_name}, got {summary.study_name}"
        )
    return optuna.load_study(study_name=study_name, storage=study_storage(db_path))


def load_or_create_materialized_study(db_path: Path, *, config: TuneConfig) -> optuna.Study:
    summary = _materialized_study_summary(db_path)
    if summary is None:
        return optuna.create_study(
            study_name=config.study.name,
            storage=study_storage(db_path),
            direction=config.objective.direction.value,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=config.tuning.sampler_seed),
            pruner=(
                optuna.pruners.MedianPruner()
                if config.tuning.enable_pruning
                else optuna.pruners.NopPruner()
            ),
        )
    if summary.study_name != config.study.name:
        raise StateLayoutError(
            "Materialized Optuna study name does not match stored manifest: "
            f"expected {config.study.name}, got {summary.study_name}"
        )
    return optuna.load_study(study_name=summary.study_name, storage=study_storage(db_path))


def _load_verified_study(
    db_path: Path,
    *,
    study_name: str | None = None,
) -> tuple[StudyManifest, optuna.Study]:
    manifest = load_study_manifest(db_path)
    if study_name is not None and manifest.study_name != study_name:
        raise StateConflictError(
            f"Study name mismatch: expected {manifest.study_name}, got {study_name}"
        )
    study = load_materialized_study(db_path, study_name=manifest.study_name)
    return manifest, study


def _materialized_study_summary(db_path: Path) -> optuna.study.StudySummary | None:
    if not _has_materialized_optuna_tables(db_path):
        return None
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        return None
    if len(summaries) != 1:
        raise StateLayoutError(
            f"Expected exactly one Optuna study in {db_path}, found {len(summaries)}"
        )
    return summaries[0]


def _has_materialized_optuna_tables(db_path: Path) -> bool:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
    except sqlite3.Error:
        return False
    table_names = {str(row[0]) for row in rows}
    return {"studies", "study_directions", "trials"} <= table_names

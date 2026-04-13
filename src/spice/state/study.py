"""Study-root state, manifest persistence, and Optuna access."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import optuna
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial, TrialState
from sqlalchemy import select

from ..config import (
    FeatureSetConfig,
    ModelConfig,
    SplitConfig,
    StudyDirection,
    TaskSpec,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TunedParameterSet,
    TuningObjective,
    TuningSpaceConfig,
)
from ..modeling.registry import (
    coerce_model_config,
    coerce_tuned_parameter_set,
    coerce_tuning_space_config,
)
from .engine import STUDY_ROOT_KIND, create_state_engine, db_url, ensure_state_db
from .schema import STUDY_TABLES, study_manifest

TRIAL_PARAMS_KEY = "spice_params"
TRIAL_BEST_EPOCH_KEY = "spice_best_epoch"
_STUDY_SAMPLER_NAME = "TPESampler"


class StudyTrialState(StrEnum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


@dataclass(frozen=True, slots=True)
class StudyManifest:
    study_id: str
    study_name: str
    chain_name: str
    dataset_id: str
    dataset_name: str
    task: TaskSpec
    feature_set: FeatureSetConfig
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    objective_metric: TuningObjective
    direction: StudyDirection
    sampler_name: str
    sampler_seed: int
    pruner_name: str
    enable_pruning: bool
    tuning_space: TuningSpaceConfig

    @property
    def task_id(self) -> str:
        return self.task.id

    @property
    def feature_set_id(self) -> str:
        return self.feature_set.id

    @property
    def model_id(self) -> str:
        return self.model.id


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


def study_storage(db_path: Path) -> RDBStorage:
    return RDBStorage(
        url=db_url(db_path),
        engine_kwargs={"connect_args": {"timeout": 5}},
    )


def manifest_from_tune_config(config: TuneConfig) -> StudyManifest:
    if config.paths.study_id is None:
        raise ValueError("study_id is required for study manifests")
    return StudyManifest(
        study_id=config.paths.study_id,
        study_name=config.study.name,
        chain_name=config.chain.name,
        dataset_id=config.paths.corpus_id,
        dataset_name=config.dataset.name,
        task=config.task,
        feature_set=config.feature_set,
        model=config.model,
        split=config.split,
        training=config.training,
        objective_metric=config.tuning.objective_metric,
        direction=config.tuning.direction,
        sampler_name=_STUDY_SAMPLER_NAME,
        sampler_seed=config.tuning.sampler_seed,
        pruner_name=_pruner_name(config.tuning.enable_pruning),
        enable_pruning=config.tuning.enable_pruning,
        tuning_space=config.tuning_space,
    )


def insert_study_manifest(db_path: Path, *, manifest: StudyManifest) -> None:
    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            existing = conn.execute(select(study_manifest.c.singleton)).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"Study manifest already exists: {db_path}")
            conn.execute(study_manifest.insert().values(**_manifest_values(manifest)))
    finally:
        engine.dispose()


def load_study_manifest(db_path: Path) -> StudyManifest:
    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(study_manifest)).mappings().first()
        if row is None:
            raise ValueError(f"Missing study manifest: {db_path}")
        model = coerce_model_config(_mapping(row["model"]))
        return StudyManifest(
            study_id=str(row["study_id"]),
            study_name=str(row["study_name"]),
            chain_name=str(row["chain_name"]),
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            task=TaskSpec.model_validate(_mapping(row["task"])),
            feature_set=FeatureSetConfig.model_validate(_mapping(row["feature_set"])),
            model=model,
            split=SplitConfig.model_validate(_mapping(row["split"])),
            training=TrainingConfig.model_validate(_mapping(row["training"])),
            objective_metric=TuningObjective(str(row["objective_metric"])),
            direction=StudyDirection(str(row["direction"])),
            sampler_name=str(row["sampler_name"]),
            sampler_seed=int(row["sampler_seed"]),
            pruner_name=str(row["pruner_name"]),
            enable_pruning=bool(row["enable_pruning"]),
            tuning_space=_coerce_study_tuning_space(row["tuning_space"], model=model),
        )
    finally:
        engine.dispose()


def load_study(db_path: Path, *, study_name: str) -> optuna.Study:
    manifest = load_study_manifest(db_path)
    if manifest.study_name != study_name:
        raise ValueError(
            f"Study name mismatch: expected {manifest.study_name}, got {study_name}"
        )
    return _load_materialized_study(db_path, manifest=manifest)


def open_tuning_study(db_path: Path, *, config: TuneConfig) -> OpenStudy:
    requested_manifest = manifest_from_tune_config(config)
    stored_manifest = _try_load_study_manifest(db_path)
    if stored_manifest is None:
        insert_study_manifest(db_path, manifest=requested_manifest)
        manifest = requested_manifest
    else:
        mismatches = diff_study_manifests(stored_manifest, requested_manifest)
        if mismatches:
            raise ValueError(
                "Existing study definition does not match current request: "
                + ", ".join(mismatches)
            )
        manifest = stored_manifest
    study = _load_or_create_materialized_study(db_path, manifest=manifest)
    existing_trial_count = len(study.trials)
    target_trial_count = config.tuning.trial_count
    if target_trial_count < existing_trial_count:
        raise ValueError(
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
        raise ValueError(
            f"Study name mismatch: expected {manifest.study_name}, got {study_name}"
        )
    study = _load_materialized_study(db_path, manifest=manifest)
    payload = study.best_trial.user_attrs.get(TRIAL_PARAMS_KEY)
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"Best tuning params are required but missing: {db_path}")
    return coerce_tuned_parameter_set(payload, model_id=manifest.model_id)


def load_study_summary(db_path: Path) -> StudySummary:
    manifest = load_study_manifest(db_path)
    study = _load_materialized_study(db_path, manifest=manifest)
    return build_study_summary(manifest, study)


def list_trial_records(db_path: Path) -> list[StudyTrialRecord]:
    manifest = load_study_manifest(db_path)
    study = _load_materialized_study(db_path, manifest=manifest)
    return [build_trial_record(trial, model_id=manifest.model_id) for trial in study.trials]


def describe_study(db_path: Path, *, detail: str | None) -> dict[str, object]:
    summary = load_study_summary(db_path)
    manifest = summary.manifest
    best_trial = summary.best_trial
    payload: dict[str, object] = {
        "study": {
            "study_id": manifest.study_id,
            "study_name": manifest.study_name,
            "chain_name": manifest.chain_name,
            "dataset_id": manifest.dataset_id,
            "dataset_name": manifest.dataset_name,
            "task_id": manifest.task_id,
            "feature_set_id": manifest.feature_set_id,
            "model_id": manifest.model_id,
            "objective_metric": manifest.objective_metric.value,
            "direction": manifest.direction.name,
            "sampler_name": manifest.sampler_name,
            "sampler_seed": manifest.sampler_seed,
            "pruner_name": manifest.pruner_name,
            "trial_count": summary.trial_counts.total,
            "best_value": None if best_trial is None else best_trial.value,
            "best_trial_number": None if best_trial is None else best_trial.number,
        }
    }
    if detail == "config":
        payload["config"] = _manifest_detail_payload(manifest)
    if detail == "trials":
        payload["trials"] = [
            {
                "number": record.number,
                "state": record.state.value,
                "value": record.value,
                "best_epoch": record.best_epoch,
                "params": record.params.model_dump(mode="json", exclude_none=True),
            }
            for record in list_trial_records(db_path)
        ]
    return payload


def record_trial_params(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr(TRIAL_PARAMS_KEY, _trial_params_payload(params))


def record_trial_best_epoch(trial: optuna.Trial, best_epoch: int) -> None:
    trial.set_user_attr(TRIAL_BEST_EPOCH_KEY, best_epoch)


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
        state=_state_from_optuna(trial.state),
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


def diff_study_manifests(stored: StudyManifest, requested: StudyManifest) -> list[str]:
    stored_payload = _study_semantics_payload(stored)
    requested_payload = _study_semantics_payload(requested)
    return [
        key
        for key in stored_payload
        if stored_payload[key] != requested_payload[key]
    ]


def validate_tuned_train_request(config: TrainConfig, *, manifest: StudyManifest) -> None:
    if config.paths.study_id is None:
        raise ValueError("study_id is required for tuned artifacts")
    stored_payload = {
        "study_name": manifest.study_name,
        "study_id": manifest.study_id,
        "chain_name": manifest.chain_name,
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "task": manifest.task.model_dump(mode="json"),
        "feature_set": manifest.feature_set.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
    }
    requested_payload = {
        "study_name": config.study.name,
        "study_id": config.paths.study_id,
        "chain_name": config.chain.name,
        "dataset_id": config.paths.corpus_id,
        "dataset_name": config.dataset.name,
        "task": config.task.model_dump(mode="json"),
        "feature_set": config.feature_set.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json", exclude_none=True),
    }
    mismatches = [
        key
        for key in stored_payload
        if stored_payload[key] != requested_payload[key]
    ]
    if mismatches:
        raise ValueError(
            "Tuned artifact request does not match study definition: "
            + ", ".join(mismatches)
        )


def _try_load_study_manifest(db_path: Path) -> StudyManifest | None:
    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(study_manifest.c.singleton)).scalar_one_or_none()
        if row is None:
            return None
    finally:
        engine.dispose()
    return load_study_manifest(db_path)


def _load_materialized_study(db_path: Path, *, manifest: StudyManifest) -> optuna.Study:
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        raise ValueError(f"Missing materialized Optuna study: {db_path}")
    if len(summaries) != 1:
        raise ValueError(f"Expected exactly one Optuna study in {db_path}, found {len(summaries)}")
    summary = summaries[0]
    if summary.study_name != manifest.study_name:
        raise ValueError(
            "Materialized Optuna study name does not match stored manifest: "
            f"expected {manifest.study_name}, got {summary.study_name}"
        )
    return optuna.load_study(study_name=manifest.study_name, storage=study_storage(db_path))


def _load_or_create_materialized_study(db_path: Path, *, manifest: StudyManifest) -> optuna.Study:
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        return optuna.create_study(
            study_name=manifest.study_name,
            storage=study_storage(db_path),
            direction=manifest.direction,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=manifest.sampler_seed),
            pruner=(
                optuna.pruners.MedianPruner()
                if manifest.enable_pruning
                else optuna.pruners.NopPruner()
            ),
        )
    return _load_materialized_study(db_path, manifest=manifest)


def _manifest_values(manifest: StudyManifest) -> dict[str, object]:
    now = int(time.time())
    return {
        "singleton": 1,
        "study_id": manifest.study_id,
        "study_name": manifest.study_name,
        "chain_name": manifest.chain_name,
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "task_id": manifest.task_id,
        "feature_set_id": manifest.feature_set_id,
        "model_id": manifest.model_id,
        "task": manifest.task.model_dump(mode="json"),
        "feature_set": manifest.feature_set.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
        "split": manifest.split.model_dump(mode="json"),
        "training": manifest.training.model_dump(mode="json"),
        "objective_metric": manifest.objective_metric.value,
        "direction": manifest.direction.value,
        "sampler_name": manifest.sampler_name,
        "sampler_seed": manifest.sampler_seed,
        "pruner_name": manifest.pruner_name,
        "enable_pruning": manifest.enable_pruning,
        "tuning_space": manifest.tuning_space.model_dump(mode="json", exclude_none=True),
        "created_at": now,
        "updated_at": now,
    }


def _manifest_detail_payload(manifest: StudyManifest) -> dict[str, object]:
    return {
        "task": manifest.task.model_dump(mode="json"),
        "feature_set": manifest.feature_set.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
        "split": manifest.split.model_dump(mode="json"),
        "training": manifest.training.model_dump(mode="json"),
        "tuning": {
            "objective_metric": manifest.objective_metric.value,
            "direction": manifest.direction.value,
            "sampler_name": manifest.sampler_name,
            "sampler_seed": manifest.sampler_seed,
            "pruner_name": manifest.pruner_name,
            "enable_pruning": manifest.enable_pruning,
        },
        "tuning_space": manifest.tuning_space.model_dump(mode="json", exclude_none=True),
    }


def _study_semantics_payload(manifest: StudyManifest) -> dict[str, object]:
    return {
        "study_name": manifest.study_name,
        "study_id": manifest.study_id,
        "chain_name": manifest.chain_name,
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "task": manifest.task.model_dump(mode="json"),
        "feature_set": manifest.feature_set.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json", exclude_none=True),
        "split": manifest.split.model_dump(mode="json"),
        "training": manifest.training.model_dump(mode="json"),
        "objective_metric": manifest.objective_metric.value,
        "direction": manifest.direction.value,
        "sampler_name": manifest.sampler_name,
        "sampler_seed": manifest.sampler_seed,
        "pruner_name": manifest.pruner_name,
        "enable_pruning": manifest.enable_pruning,
        "tuning_space": manifest.tuning_space.model_dump(mode="json", exclude_none=True),
    }


def _coerce_study_tuning_space(payload: object, *, model: ModelConfig) -> TuningSpaceConfig:
    if not isinstance(payload, dict):
        raise TypeError("Study tuning_space payload must be a mapping")
    tuning_space = coerce_tuning_space_config(payload, model_config=model)
    if tuning_space is None:
        raise ValueError("Study tuning_space payload is required")
    return tuning_space


def _trial_params_payload(params: TunedParameterSet) -> dict[str, object]:
    payload = params.model_dump(mode="json", exclude_none=True)
    if not isinstance(payload, dict):
        raise TypeError("TunedParameterSet did not serialize to a mapping payload")
    return payload


def _pruner_name(enable_pruning: bool) -> str:
    return "MedianPruner" if enable_pruning else "NopPruner"


def _state_from_optuna(state: TrialState) -> StudyTrialState:
    return StudyTrialState(state.name)


def _mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Expected mapping payload")
    return dict(payload)

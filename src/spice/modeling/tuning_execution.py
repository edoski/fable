"""Tuning study execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config.models import TuneConfig, TunedParameterSet
from ..core.errors import StateConflictError
from ..corpus.metadata import DatasetManifest
from ..storage.study_manifest import (
    diff_study_manifests,
    insert_study_manifest,
    manifest_from_tune_config,
    try_load_study_manifest,
)
from ..storage.study_models import (
    TRIAL_BEST_EPOCH_KEY,
    TRIAL_PARAMS_KEY,
    StudyManifest,
    StudySummary,
    best_epoch_from_trial,
    build_study_summary,
    trial_params_payload,
)
from ..storage.study_optuna import load_or_create_materialized_study
from .persisted_training import run_trial_training
from .pipeline import build_trial_training_spec
from .tuned_config import sample_tuned_parameters
from .tuning import apply_tuned_parameters

if TYPE_CHECKING:
    from ..storage.workflow_roots import TuneWorkflowRoots


@dataclass(frozen=True, slots=True)
class OpenTuningExecution:
    manifest: StudyManifest
    study: optuna.Study
    existing_trial_count: int
    target_trial_count: int
    remaining_trial_count: int


@dataclass(frozen=True, slots=True)
class TuningTrialProgress:
    number: int
    total_trials: int
    state: str
    value: float | None
    best_epoch: int | None


@dataclass(frozen=True, slots=True)
class TuningBestProgress:
    trial_number: int
    value: float


@dataclass(frozen=True, slots=True)
class TuningExecutionCallbacks:
    on_resume: Callable[[int, int], None] | None = None
    on_study_start: Callable[[int], None] | None = None
    on_trial_complete: Callable[[TuningTrialProgress], None] | None = None
    on_best_improved: Callable[[TuningBestProgress], None] | None = None


def open_tuning_execution(
    config: TuneConfig,
    *,
    roots: TuneWorkflowRoots,
    corpus_manifest: DatasetManifest,
) -> OpenTuningExecution:
    requested_manifest = manifest_from_tune_config(
        config,
        corpus=roots.corpus,
        study=roots.study,
        corpus_manifest=corpus_manifest,
    )
    db_path = roots.study.state_db_path
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
    return OpenTuningExecution(
        manifest=manifest,
        study=materialized_study,
        existing_trial_count=existing_trial_count,
        target_trial_count=target_trial_count,
        remaining_trial_count=target_trial_count - existing_trial_count,
    )


def run_tuning_execution(
    opened: OpenTuningExecution,
    *,
    config: TuneConfig,
    roots: TuneWorkflowRoots,
    corpus_manifest: DatasetManifest,
    callbacks: TuningExecutionCallbacks | None = None,
) -> StudySummary:
    active_callbacks = callbacks or TuningExecutionCallbacks()
    with _optuna_warning_verbosity():
        if opened.existing_trial_count and active_callbacks.on_resume is not None:
            active_callbacks.on_resume(
                opened.existing_trial_count,
                opened.target_trial_count,
            )

        best_trial_number = _current_best_trial_number(opened.study)

        def on_trial_complete(active_study: optuna.Study, frozen_trial: FrozenTrial) -> None:
            nonlocal best_trial_number
            if active_callbacks.on_trial_complete is not None:
                active_callbacks.on_trial_complete(
                    TuningTrialProgress(
                        number=frozen_trial.number,
                        total_trials=opened.target_trial_count,
                        state=frozen_trial.state.name,
                        value=frozen_trial.value,
                        best_epoch=best_epoch_from_trial(frozen_trial),
                    )
                )
            if frozen_trial.state != TrialState.COMPLETE:
                return
            try:
                study_best = active_study.best_trial
            except ValueError:
                return
            if study_best.number == best_trial_number:
                return
            best_trial_number = study_best.number
            if (
                study_best.value is not None
                and active_callbacks.on_best_improved is not None
            ):
                active_callbacks.on_best_improved(
                    TuningBestProgress(
                        trial_number=study_best.number,
                        value=study_best.value,
                    )
                )

        if opened.remaining_trial_count > 0:
            if active_callbacks.on_study_start is not None:
                active_callbacks.on_study_start(opened.remaining_trial_count)
            opened.study.optimize(
                lambda trial: _trial_objective(
                    config,
                    trial,
                    roots=roots,
                    corpus_manifest=corpus_manifest,
                ),
                n_trials=opened.remaining_trial_count,
                timeout=config.tuning.timeout_seconds,
                callbacks=[on_trial_complete],
            )
        return build_study_summary(opened.manifest, opened.study)


def _current_best_trial_number(study: optuna.Study) -> int | None:
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        return None
    try:
        return study.best_trial.number
    except ValueError:
        return None


@contextmanager
def _optuna_warning_verbosity() -> Iterator[None]:
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous_verbosity)


def _trial_objective(
    base_config: TuneConfig,
    trial: optuna.Trial,
    *,
    roots: TuneWorkflowRoots,
    corpus_manifest: DatasetManifest,
) -> float:
    assert base_config.tuning_space is not None
    params = sample_tuned_parameters(trial, tuning_space=base_config.tuning_space)
    _record_trial_params(trial, params)
    config = apply_tuned_parameters(base_config, params)

    spec = build_trial_training_spec(
        config,
        corpus=roots.corpus,
        study=roots.study,
        corpus_manifest=corpus_manifest,
    )
    history_block_path = roots.corpus.history_dir
    trial_run = run_trial_training(history_block_path, spec=spec)
    metric_value = trial_run.summary.runtime.best_objective_value
    best_epoch = trial_run.training_run.training_result.best_epoch
    _record_trial_best_epoch(trial, best_epoch)
    if config.tuning.enable_pruning:
        trial.report(metric_value, step=best_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return metric_value


def _record_trial_params(trial: optuna.Trial, params: TunedParameterSet) -> None:
    trial.set_user_attr(TRIAL_PARAMS_KEY, trial_params_payload(params))


def _record_trial_best_epoch(trial: optuna.Trial, best_epoch: int) -> None:
    trial.set_user_attr(TRIAL_BEST_EPOCH_KEY, best_epoch)

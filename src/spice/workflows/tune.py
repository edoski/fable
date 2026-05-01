"""Optuna tuning workflow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config.models import TuneConfig, TunedParameterSet, TunedProblemParams
from ..core.rendering import metric_string
from ..core.reporting import Reporter
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..modeling.persisted_training import run_persisted_training
from ..modeling.pipeline import build_trial_training_spec
from ..modeling.tuned_config import sample_tuned_parameters
from ..modeling.tuning import apply_tuned_parameters
from ..storage.study_models import best_epoch_from_trial, build_study_summary
from ..storage.study_optuna import (
    open_tuning_study,
    record_trial_best_epoch,
    record_trial_params,
)
from ..storage.study_render import study_result_fields
from ..storage.workflow_roots import TuneWorkflowRoots, resolve_tune_roots


def _workflow_facts(config: TuneConfig, roots: TuneWorkflowRoots) -> list[tuple[str, str]]:
    return [
        ("dataset", roots.corpus.dataset_name),
        ("dataset_id", roots.corpus.dataset_id),
        ("chain", roots.corpus.chain_name),
        ("problem", config.problem.id),
        ("features", config.features.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("study", roots.study.study_name),
        ("study_id", roots.study.study_id),
        ("trials", str(config.tuning.trial_count)),
    ]


def _trial_work_dir(study_root: Path, trial_number: int) -> TemporaryDirectory[str]:
    return TemporaryDirectory(
        dir=study_root.parent,
        prefix=f".trial-{trial_number:03d}.",
    )


@contextmanager
def _optuna_warning_verbosity() -> Iterator[None]:
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous_verbosity)


def _trial_message(
    trial: FrozenTrial,
    *,
    total_trials: int,
) -> str:
    parts = [f"trial {trial.number + 1}/{total_trials}"]
    if trial.state == TrialState.COMPLETE:
        parts.append("complete")
        if trial.value is not None:
            parts.append(f"value={metric_string(trial.value)}")
        best_epoch = best_epoch_from_trial(trial)
        if best_epoch is not None:
            parts.append(f"best_epoch={best_epoch}")
        return " ".join(parts)
    if trial.state == TrialState.PRUNED:
        parts.append("pruned")
        return " ".join(parts)
    parts.append("failed")
    return " ".join(parts)


def _objective(
    base_config: TuneConfig,
    trial: optuna.Trial,
    *,
    roots: TuneWorkflowRoots,
    corpus_manifest,
) -> float:
    assert base_config.tuning_space is not None
    params = sample_tuned_parameters(trial, tuning_space=base_config.tuning_space)
    record_trial_params(trial, params)
    config = apply_tuned_parameters(base_config, params)

    spec = build_trial_training_spec(
        config,
        corpus=roots.corpus,
        study=roots.study,
        corpus_manifest=corpus_manifest,
    )
    history_block_path = roots.corpus.history_dir
    with _trial_work_dir(roots.study.root_path, trial.number) as temp_dir_name:
        artifact_dir = Path(temp_dir_name)
        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            persist_artifact=False,
        )
    metric_value = persisted.summary.runtime.best_objective_value
    record_trial_best_epoch(trial, persisted.training_run.training_result.best_epoch)
    if config.tuning.enable_pruning:
        trial.report(metric_value, step=persisted.training_run.training_result.best_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return metric_value


def _coverage_spec(config: TuneConfig, *, roots: TuneWorkflowRoots, corpus_manifest):
    if (
        config.tuning_space.problem is None
        or config.tuning_space.problem.lookback_seconds is None
    ):
        return build_trial_training_spec(
            config,
            corpus=roots.corpus,
            study=roots.study,
            corpus_manifest=corpus_manifest,
        )
    return build_trial_training_spec(
        apply_tuned_parameters(
            config,
            TunedParameterSet(
                problem=TunedProblemParams(
                    lookback_seconds=max(config.tuning_space.problem.lookback_seconds)
                )
            ),
        ),
        corpus=roots.corpus,
        study=roots.study,
        corpus_manifest=corpus_manifest,
    )


def run(config: TuneConfig, *, reporter: Reporter | None = None) -> None:
    active_reporter = reporter or Reporter()
    roots = resolve_tune_roots(config)
    corpus_manifest = roots.corpus.load_manifest()
    active_reporter.header("tune", _workflow_facts(config, roots))
    spec = _coverage_spec(config, roots=roots, corpus_manifest=corpus_manifest)
    validate_corpus_coverage(
        corpus_manifest,
        contract=spec.problem_contract,
        feature_contract=spec.feature_contract,
        requirement=training_coverage_requirement(spec.problem_contract),
    )

    with _optuna_warning_verbosity():
        study_access = open_tuning_study(
            roots.study,
            config=config,
            corpus=roots.corpus,
            corpus_manifest=corpus_manifest,
        )
        roots.study.reindex()
        study = study_access.study
        if study_access.existing_trial_count:
            active_reporter.milestone(
                "resume "
                f"trials={study_access.existing_trial_count}/"
                f"{study_access.target_trial_count}"
            )

        existing_best = next(
            (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
            None,
        )
        best_trial_number = None if existing_best is None else study.best_trial.number

        def on_trial_complete(active_study: optuna.Study, frozen_trial: FrozenTrial) -> None:
            nonlocal best_trial_number
            active_reporter.milestone(
                _trial_message(
                    frozen_trial,
                    total_trials=study_access.target_trial_count,
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
            if study_best.value is not None:
                active_reporter.milestone(
                    "best improved "
                    f"trial={study_best.number + 1} "
                    f"value={metric_string(study_best.value)}"
                )

        if study_access.remaining_trial_count > 0:
            active_reporter.milestone(f"study started trials={study_access.remaining_trial_count}")
            study.optimize(
                lambda trial: _objective(
                    config,
                    trial,
                    roots=roots,
                    corpus_manifest=corpus_manifest,
                ),
                n_trials=study_access.remaining_trial_count,
                timeout=config.tuning.timeout_seconds,
                callbacks=[on_trial_complete],
            )
        summary = build_study_summary(study_access.manifest, study)
    roots.study.reindex()
    active_reporter.result("tune", study_result_fields(summary))

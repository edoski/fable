"""Optuna tuning workflow."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config import TuneConfig
from ..core.reporting import Reporter, StageMetricDescriptor
from ..core.runtime import ConsoleRuntime
from ..modeling.execution import run_persisted_training
from ..modeling.families.registry import sample_tuned_parameters
from ..modeling.pipeline import TrainingStageReporters, build_training_spec
from ..modeling.tuning import apply_tuned_parameters
from ..prediction import compile_prediction_contract
from ..storage.catalog import upsert_study_record
from ..storage.study import (
    build_study_summary,
    open_tuning_study,
    record_trial_best_epoch,
    record_trial_params,
    study_summary_sections,
)
from ._shared import managed_workflow

_TRIAL_STAGE_METRICS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="epoch", label="epoch", width=7),
)


def _trial_status_message(trial: FrozenTrial) -> str:
    if trial.state == TrialState.COMPLETE:
        assert trial.value is not None
        return f"trial {trial.number + 1} complete: value={trial.value:.4f}"
    if trial.state == TrialState.PRUNED:
        return f"trial {trial.number + 1} pruned"
    return f"trial {trial.number + 1} failed"


def _trial_stage_status(trial: FrozenTrial) -> str:
    if trial.state == TrialState.FAIL:
        return "failed"
    return "done"


def _workflow_facts(config: TuneConfig) -> list[tuple[str, str]]:
    return [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("feature set", config.feature_set.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("study", config.study.name),
    ]


def _trial_work_dir(study_root: Path, trial_number: int) -> TemporaryDirectory[str]:
    return TemporaryDirectory(
        dir=study_root.parent,
        prefix=f".trial-{trial_number:03d}.",
    )


def _objective(
    base_config: TuneConfig,
    trial: optuna.Trial,
    *,
    study_root: Path,
    runtime: ConsoleRuntime,
    trial_reporter: Reporter,
) -> float:
    assert base_config.tuning_space is not None
    params = sample_tuned_parameters(trial, tuning_space=base_config.tuning_space)
    record_trial_params(trial, params)
    config = cast(TuneConfig, apply_tuned_parameters(base_config, params))

    spec = build_training_spec(config)
    history_block_path = config.paths.history_dir
    runtime.set_stage_state(
        "trial",
        status="running",
        message=f"trial {trial.number + 1}/{base_config.tuning.trial_count}",
    )
    with _trial_work_dir(study_root, trial.number) as temp_dir_name:
        artifact_dir = Path(temp_dir_name)
        with managed_workflow(
            config,
            run_name=f"trial-{trial.number:03d}",
            runtime=runtime,
            reporter=trial_reporter,
            nested=True,
        ) as session:
            persisted = run_persisted_training(
                history_block_path,
                spec=spec,
                artifact_dir=artifact_dir,
                stage_reporters=TrainingStageReporters.shared(trial_reporter),
                write_reporter=trial_reporter,
                reporter=session.reporter,
                persist_artifact=False,
            )
    metric_value = spec.prediction_contract.optimization_value(
        persisted.best_validation_metrics
    )
    record_trial_best_epoch(trial, persisted.training_run.training_result.best_epoch)
    if config.tuning.enable_pruning:
        trial.report(metric_value, step=persisted.training_run.training_result.best_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return metric_value


def run(config: TuneConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        config,
        run_name=(
            "study-"
            f"{config.chain.name}-{config.model.id}-"
            f"{config.problem.id}"
        ),
        reporter=reporter,
    ) as session:
        session.runtime.configure_workflow("tune", _workflow_facts(config))
        study_root = config.paths.study_root
        study_state_db = config.paths.study_state_db
        study_id = config.paths.study_id
        if study_root is None or study_state_db is None or study_id is None:
            raise ValueError("tuning workflow requires study output paths")
        study_reporter = session.runtime.stage_reporter(
            "study",
            label="study",
            total=config.tuning.trial_count,
            unit="trials",
        )
        prediction_contract = compile_prediction_contract(
            prediction_id=config.prediction.id,
            family_config=config.prediction.family,
        )
        trial_reporter = session.runtime.stage_reporter(
            "trial",
            label="trial",
            metric_descriptors=(
                *_TRIAL_STAGE_METRICS,
                *prediction_contract.progress_metric_descriptors,
            ),
        )
        study_access = open_tuning_study(study_state_db, config=config)
        study_task = study_reporter.start_task(
            "tune study",
            total=study_access.target_trial_count,
            unit="trials",
        )
        if study_access.existing_trial_count:
            study_reporter.update_task(
                study_task,
                completed=study_access.existing_trial_count,
                message=f"resuming from trial {study_access.existing_trial_count}",
            )
        study = study_access.study

        def on_trial_complete(active_study: optuna.Study, frozen_trial: FrozenTrial) -> None:
            del active_study
            study_reporter.update_task(
                study_task,
                completed=frozen_trial.number + 1,
                message=_trial_status_message(frozen_trial),
            )
            session.runtime.set_stage_state(
                "trial",
                status=_trial_stage_status(frozen_trial),
                message=_trial_status_message(frozen_trial),
            )

        if study_access.remaining_trial_count == 0:
            session.runtime.set_stage_state(
                "trial",
                status="done",
                message="study already at requested trial count",
            )
        else:
            with session.runtime.optuna_logging():
                study.optimize(
                    lambda trial: _objective(
                        config,
                        trial,
                        study_root=study_root,
                        runtime=session.runtime,
                        trial_reporter=trial_reporter,
                    ),
                    n_trials=study_access.remaining_trial_count,
                    timeout=config.tuning.timeout_seconds,
                    callbacks=[on_trial_complete],
                )
        completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if completed_trials:
            study_reporter.finish_task(
                study_task,
                message=f"best_value={study.best_value:.4f}",
            )
        else:
            study_reporter.finish_task(study_task, message="no successful trials")
        summary = build_study_summary(study_access.manifest, study)
        upsert_study_record(
            config.paths.catalog_db,
            study_id=study_id,
            study_name=config.study.name,
            dataset_id=config.paths.corpus_id,
            dataset_name=config.dataset.name,
            chain_name=config.chain.name,
            feature_set_id=config.feature_set.id,
            prediction_id=config.prediction.id,
            model_id=config.model.id,
            problem_id=config.problem.id,
            root_path=study_root,
            state_db_path=study_state_db,
        )
        session.runtime.log_sectioned_summary(
            "tuning summary",
            study_summary_sections(summary),
        )

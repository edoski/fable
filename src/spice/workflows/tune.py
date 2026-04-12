"""Optuna tuning workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
from optuna.trial import FrozenTrial, TrialState

from ..config import TuneConfig
from ..core.console import ConsoleRuntime, Reporter
from ..core.files import remove_path
from ..modeling.execution import run_persisted_training
from ..modeling.pipeline import TrainingStageReporters
from ..modeling.registry import sample_tuned_parameters
from ..state.study import create_or_load_study
from ._shared import abort_cleanup, build_training_spec, epoch_metrics_to_dict, managed_workflow
from ._tuning import (
    apply_tuned_parameters,
    build_study_summary,
    flatten_tuned_parameters,
    freeze_best_epoch_for_trial,
    freeze_tuned_parameters_for_trial,
)


def _format_best_params(params) -> str:
    flattened = flatten_tuned_parameters(params)
    if not flattened:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in flattened.items())


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
        ("dataset", config.dataset.id),
        ("chain", config.chain.name),
        ("task", config.task.id),
        ("feature set", config.feature_set.id),
        ("model", config.model.id),
        ("study", config.study.id),
    ]


def _trial_work_dir(artifact_root: Path, trial_number: int) -> TemporaryDirectory[str]:
    return TemporaryDirectory(
        dir=artifact_root.parent,
        prefix=f".trial-{trial_number:03d}.",
    )


def _objective(
    base_config: TuneConfig,
    trial: optuna.Trial,
    *,
    artifact_root: Path,
    runtime: ConsoleRuntime,
    trial_reporter: Reporter,
) -> float:
    assert base_config.tuning_space is not None
    params = sample_tuned_parameters(trial, tuning_space=base_config.tuning_space)
    freeze_tuned_parameters_for_trial(trial, params)
    config = apply_tuned_parameters(base_config, params)

    spec = build_training_spec(config)
    history_block_path = config.paths.history_dir
    runtime.set_stage_state(
        "trial",
        status="running",
        message=f"trial {trial.number + 1}/{base_config.tuning.trial_count}",
    )
    with _trial_work_dir(artifact_root, trial.number) as temp_dir_name:
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
    metric_map = epoch_metrics_to_dict(persisted.best_validation_metrics)
    metric_value = metric_map[config.tuning.objective_metric.metric_name]
    freeze_best_epoch_for_trial(trial, persisted.training_run.training_result.best_epoch)
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
            f"{config.task.id}"
        ),
        reporter=reporter,
    ) as session:
        session.runtime.configure_workflow("tune", _workflow_facts(config))
        artifact_root = config.paths.artifact_root
        study_state_db = config.paths.study_state_db
        if artifact_root is None or study_state_db is None:
            raise ValueError("tuning workflow requires study output paths")
        study_reporter = session.runtime.stage_reporter(
            "study",
            label="study",
            total=config.tuning.trial_count,
            unit="trials",
        )
        trial_reporter = session.runtime.stage_reporter("trial", label="trial")
        with abort_cleanup(
            session.reporter,
            label="tune",
            cleanup=lambda: remove_path(artifact_root),
        ):
            if artifact_root.exists():
                shutil.rmtree(artifact_root)
            study_task = study_reporter.start_task(
                "tune study",
                total=config.tuning.trial_count,
                unit="trials",
            )
            study = create_or_load_study(study_state_db, config=config)

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

            with session.runtime.optuna_logging():
                study.optimize(
                    lambda trial: _objective(
                        config,
                        trial,
                        artifact_root=artifact_root,
                        runtime=session.runtime,
                        trial_reporter=trial_reporter,
                    ),
                    n_trials=config.tuning.trial_count,
                    timeout=config.tuning.timeout_seconds,
                    callbacks=[on_trial_complete],
                )
            completed_trials = [
                trial for trial in study.trials if trial.state == TrialState.COMPLETE
            ]
            if completed_trials:
                study_reporter.finish_task(
                    study_task,
                    message=f"best_value={study.best_value:.4f}",
                )
            else:
                study_reporter.finish_task(study_task, message="no successful trials")
            summary = build_study_summary(config, study)
            best_trial = summary.best_trial
            session.runtime.log_sectioned_summary(
                "tuning summary",
                [
                    (
                        "study",
                        [
                            ("id", summary.study.id),
                            ("chain", summary.chain),
                            ("task", summary.task_id),
                            ("model", summary.model_id),
                            ("trials", str(summary.trial_counts.total)),
                            ("state", str(study_state_db)),
                        ],
                    ),
                    (
                        "best trial",
                        [
                            ("objective", summary.objective_metric.value),
                            (
                                "value",
                                "n/a"
                                if best_trial is None or best_trial.value is None
                                else f"{best_trial.value:.4f}",
                            ),
                            (
                                "trial",
                                "n/a" if best_trial is None else str(best_trial.number + 1),
                            ),
                            (
                                "params",
                                "n/a"
                                if best_trial is None
                                else _format_best_params(best_trial.params),
                            ),
                        ],
                    ),
                ],
            )

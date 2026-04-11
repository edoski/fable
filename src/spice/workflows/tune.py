"""Hydra entrypoint for Optuna studies."""

from __future__ import annotations

import shutil

import hydra
import optuna
from omegaconf import DictConfig
from optuna.trial import FrozenTrial, TrialState

from ..core.config import ExperimentConfig, coerce_config
from ..core.console import ConsoleRuntime, Reporter
from ..core.json import write_json
from ..core.tracking import log_artifacts
from ..modeling.execution import run_persisted_training
from ._shared import (
    build_training_spec,
    epoch_metrics_to_dict,
    managed_workflow,
    trial_artifact_dir,
)
from ._tuning import (
    apply_tuned_parameters,
    build_best_params_report,
    build_study_report,
    build_trial_record,
    flatten_tuned_parameters,
    freeze_tuned_parameters_for_trial,
    sample_tuned_parameters,
)


def _trial_status_message(trial: FrozenTrial) -> str:
    if trial.state == TrialState.COMPLETE:
        assert trial.value is not None
        return f"trial {trial.number + 1} complete: value={trial.value:.4f}"
    if trial.state == TrialState.PRUNED:
        return f"trial {trial.number + 1} pruned"
    return f"trial {trial.number + 1} failed"


def _objective(
    base_config: ExperimentConfig,
    trial: optuna.Trial,
    *,
    runtime: ConsoleRuntime,
    reporter: Reporter,
) -> float:
    params = sample_tuned_parameters(trial, base_config.tuning.search_space)
    freeze_tuned_parameters_for_trial(trial, params)
    config = apply_tuned_parameters(base_config, params)

    spec = build_training_spec(config)
    artifact_dir = trial_artifact_dir(config, trial.number)
    history_block_path = config.paths.history_dir
    with managed_workflow(
        config,
        run_name=f"trial-{trial.number:03d}",
        runtime=runtime,
        reporter=reporter,
        nested=True,
    ) as session:
        if session.tracking_enabled:
            import mlflow

            mlflow.log_params(
                {
                    f"trial.{key}": str(value)
                    for key, value in flatten_tuned_parameters(params).items()
                }
            )

        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            report_path=artifact_dir / "train_report.json",
            reporter=session.reporter,
            runtime=session.runtime,
        )
        metric_map = epoch_metrics_to_dict(persisted.best_validation_metrics)
        metric_value = metric_map[config.tuning.objective_metric.metric_name]
        trial.set_user_attr("best_epoch", persisted.training_run.training_result.best_epoch)
        trial.set_user_attr("artifact_dir", str(artifact_dir))
        if config.tuning.enable_pruning:
            trial.report(metric_value, step=persisted.training_run.training_result.best_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if session.tracking_enabled:
            import mlflow

            mlflow.log_metrics({f"trial.{key}": value for key, value in metric_map.items()})
            log_artifacts(persisted.artifact_paths)
        return metric_value


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    with managed_workflow(
        config,
        run_name=(
            "study-"
            f"{config.chain.name.value}-{config.model.family.value}-"
            f"{config.dataset.temporal.max_delay_seconds}s"
        ),
        reporter=reporter,
    ) as session:
        tuning_root = config.paths.tuning_root
        if tuning_root.exists():
            shutil.rmtree(tuning_root)
        study_task = session.reporter.start_task(
            "tune study",
            total=config.tuning.trial_count,
            unit="trials",
        )

        def on_trial_complete(study: optuna.Study, frozen_trial: FrozenTrial) -> None:
            session.reporter.update_task(
                study_task,
                completed=frozen_trial.number + 1,
                message=_trial_status_message(frozen_trial),
            )

        with session.runtime.optuna_logging():
            study = optuna.create_study(
                study_name=config.tuning.study_name,
                direction=config.tuning.direction,
                pruner=(
                    optuna.pruners.MedianPruner()
                    if config.tuning.enable_pruning
                    else optuna.pruners.NopPruner()
                ),
                sampler=optuna.samplers.TPESampler(seed=config.tuning.sampler_seed),
            )
            study.optimize(
                lambda trial: _objective(
                    config,
                    trial,
                    runtime=session.runtime,
                    reporter=session.reporter,
                ),
                n_trials=config.tuning.trial_count,
                timeout=config.tuning.timeout_seconds,
                callbacks=[on_trial_complete],
            )
        study_path = tuning_root / "study.json"
        trials_path = tuning_root / "trials.json"
        best_params_path = config.paths.tuning_best_params_path
        study_report = build_study_report(config, study)
        trial_records = [build_trial_record(trial) for trial in study.trials]
        write_task = session.reporter.start_task("write tuning summary")
        write_json(study_path, study_report)
        write_json(trials_path, tuple(trial_records))
        write_json(best_params_path, build_best_params_report(config, study))
        session.reporter.finish_task(write_task, message=str(best_params_path))
        completed_trials = [
            trial for trial in study.trials if trial.state == TrialState.COMPLETE
        ]
        if completed_trials:
            session.reporter.finish_task(study_task, message=f"best_value={study.best_value:.4f}")
        else:
            session.reporter.finish_task(study_task, message="no successful trials")
        if session.tracking_enabled:
            import mlflow

            metrics = {"study.trial_count": float(len(study.trials))}
            if completed_trials:
                best_trial = study_report.best_trial
                assert best_trial is not None
                assert best_trial.value is not None
                metrics["study.best_value"] = best_trial.value
                mlflow.log_params(
                    {
                        f"study.best_param.{key}": str(value)
                        for key, value in flatten_tuned_parameters(best_trial.params).items()
                    }
                )
            mlflow.log_metrics(metrics)
            log_artifacts([study_path, trials_path, best_params_path])


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="tune"))


if __name__ == "__main__":
    main()

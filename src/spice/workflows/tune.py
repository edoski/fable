"""Hydra entrypoint for Optuna studies."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import hydra
import optuna
from omegaconf import DictConfig
from optuna.trial import FrozenTrial, TrialState

from ..core.config import ExperimentConfig, coerce_config, revalidate_config
from ..core.console import ConsoleRuntime, Reporter
from ..core.json import write_json
from ..core.tracking import log_artifacts
from ..modeling.execution import run_persisted_training
from ._shared import (
    build_training_spec,
    clone_config,
    epoch_metrics_to_dict,
    managed_workflow,
    set_nested_attr,
    trial_artifact_dir,
)


def _trial_record(trial: FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "artifact_dir": trial.user_attrs.get("artifact_dir"),
        "started_at": (
            trial.datetime_start.isoformat()
            if trial.datetime_start is not None
            else None
        ),
        "completed_at": (
            trial.datetime_complete.isoformat()
            if trial.datetime_complete is not None
            else None
        ),
    }


def _study_summary(config: ExperimentConfig, study: optuna.Study) -> dict[str, Any]:
    completed_trials = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    pruned_trials = [
        trial for trial in study.trials if trial.state == TrialState.PRUNED
    ]
    failed_trials = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if completed_trials else None

    return {
        "kind": "tuning_study",
        "study_name": config.tuning.study_name,
        "chain": config.chain.name.value,
        "dataset_id": config.dataset.id,
        "family": config.model.family.value,
        "max_delay_seconds": config.dataset.temporal.max_delay_seconds,
        "lookback_seconds": config.dataset.temporal.lookback_seconds,
        "anchor_count": config.dataset.sampling.anchor_count,
        "objective_metric": config.tuning.objective_metric,
        "direction": config.tuning.direction,
        "trial_count_requested": config.tuning.trial_count,
        "timeout_seconds": config.tuning.timeout_seconds,
        "sampler": "TPESampler",
        "sampler_seed": config.tuning.sampler_seed,
        "pruner": "MedianPruner" if config.tuning.enable_pruning else "NopPruner",
        "search_space": config.tuning.search_space,
        "trial_counts": {
            "total": len(study.trials),
            "complete": len(completed_trials),
            "pruned": len(pruned_trials),
            "failed": len(failed_trials),
        },
        "best_trial": (
            None
            if best_trial is None
            else {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": dict(best_trial.params),
                "best_epoch": best_trial.user_attrs.get("best_epoch"),
                "artifact_dir": best_trial.user_attrs.get("artifact_dir"),
            }
        ),
    }


def _best_params_summary(config: ExperimentConfig, study: optuna.Study) -> dict[str, Any]:
    completed_trials = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Optuna study completed without any successful trials")
    best_trial = study.best_trial
    return {
        "kind": "tuning_best_params",
        "study_name": config.tuning.study_name,
        "chain": config.chain.name.value,
        "dataset_id": config.dataset.id,
        "family": config.model.family.value,
        "max_delay_seconds": config.dataset.temporal.max_delay_seconds,
        "lookback_seconds": config.dataset.temporal.lookback_seconds,
        "anchor_count": config.dataset.sampling.anchor_count,
        "objective_metric": config.tuning.objective_metric,
        "direction": config.tuning.direction,
        "trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "best_epoch": best_trial.user_attrs.get("best_epoch"),
            "artifact_dir": best_trial.user_attrs.get("artifact_dir"),
        },
        "params": dict(best_trial.params),
    }


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
    config = clone_config(base_config)
    for path, candidates in config.tuning.search_space.items():
        first = candidates[0]
        if isinstance(first, str):
            value = trial.suggest_categorical(path, list(candidates))
        elif isinstance(first, int):
            value = trial.suggest_categorical(path, list(candidates))
        else:
            value = trial.suggest_categorical(path, list(candidates))
        set_nested_attr(config, path, value)
    config = revalidate_config(config)

    spec = build_training_spec(config)
    artifact_dir = trial_artifact_dir(config, trial.number)
    history_block_path = Path(config.paths.history_dir)
    with managed_workflow(
        config,
        run_name=f"trial-{trial.number:03d}",
        runtime=runtime,
        reporter=reporter,
        nested=True,
    ) as session:
        if session.tracking_enabled:
            import mlflow

            mlflow.log_params({f"trial.{key}": str(value) for key, value in trial.params.items()})

        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            report_path=artifact_dir / "train_report.json",
            reporter=session.reporter,
            runtime=session.runtime,
        )
        metric_map = epoch_metrics_to_dict(persisted.best_validation_metrics)
        metric_value = metric_map[config.tuning.objective_metric.removeprefix("validation_")]
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
        tuning_root = Path(config.paths.tuning_root)
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
        best_params_path = Path(config.paths.tuning_best_params_path)
        write_task = session.reporter.start_task("write tuning summary")
        write_json(study_path, _study_summary(config, study))
        write_json(trials_path, [_trial_record(trial) for trial in study.trials])
        write_json(best_params_path, _best_params_summary(config, study))
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
                metrics["study.best_value"] = study.best_value
                mlflow.log_params(
                    {
                        f"study.best_param.{key}": str(value)
                        for key, value in study.best_params.items()
                    }
                )
            mlflow.log_metrics(metrics)
            log_artifacts([study_path, trials_path, best_params_path])


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="tune"))


if __name__ == "__main__":
    main()

"""Hydra entrypoint for Optuna studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
import optuna
from omegaconf import DictConfig
from optuna.trial import FrozenTrial, TrialState

from ..core.config import ExperimentConfig, coerce_config, validate_config
from ..core.constants import ARTIFACT_MANIFEST_FILENAME, MODEL_STATE_FILENAME
from ..core.tracking import configure_mlflow, log_artifacts, log_config
from ..modeling.artifacts import build_training_artifact_manifest, write_training_artifact
from ..modeling.pipeline import run_training
from ..modeling.reporting import build_training_run_report, write_json_report
from ._shared import (
    build_training_spec,
    clone_config,
    epoch_metrics_to_dict,
    set_nested_attr,
    start_run_if_enabled,
    trial_artifact_dir,
)


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        "family": config.model.family.value,
        "max_delay_seconds": config.max_delay_seconds,
        "lookback_seconds": config.lookback_seconds,
        "target_anchor_count": config.target_anchor_count,
        "metric_name": config.tuning.metric_name,
        "direction": config.tuning.direction,
        "n_trials_requested": config.tuning.n_trials,
        "timeout_seconds": config.tuning.timeout_seconds,
        "sampler": "TPESampler",
        "sampler_seed": config.tuning.sampler_seed,
        "pruner": "MedianPruner" if config.tuning.prune else "NopPruner",
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


def _objective(base_config, trial: optuna.Trial) -> float:
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
    validate_config(config)

    spec = build_training_spec(config)
    artifact_dir = trial_artifact_dir(config, trial.number)
    history_block_path = Path(config.paths.enriched_history_dir)
    run_context = start_run_if_enabled(
        config,
        run_name=f"trial-{trial.number:03d}",
        nested=True,
    )
    try:
        if run_context is not None:
            run_context.__enter__()
            log_config(config)
            mlflow.log_params({f"trial.{key}": str(value) for key, value in trial.params.items()})

        result = run_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_dir,
            reporter=None,
        )
        manifest = build_training_artifact_manifest(result.prepared, spec=spec)
        write_training_artifact(artifact_dir, manifest=manifest, model=result.model)
        report = build_training_run_report(
            result,
            target_anchor_count=config.target_anchor_count,
            max_delay_seconds=config.max_delay_seconds,
            lookback_seconds=config.lookback_seconds,
            chain_name=config.chain.name.value,
            family=config.model.family.value,
            block_time_seconds=config.chain.block_time_seconds,
            manifest=manifest,
            prepared=result.prepared,
            artifact_dir=artifact_dir,
            history_block_path=history_block_path,
            device_requested=config.training.device,
        )
        report_path = artifact_dir / "train_report.json"
        write_json_report(report_path, report)
        best_metrics = result.training_result.validation_history[
            result.training_result.best_epoch - 1
        ]
        metric_map = epoch_metrics_to_dict(best_metrics)
        metric_value = metric_map[config.tuning.metric_name.removeprefix("validation_")]
        trial.set_user_attr("best_epoch", result.training_result.best_epoch)
        trial.set_user_attr("artifact_dir", str(artifact_dir))
        if config.tuning.prune:
            trial.report(metric_value, step=result.training_result.best_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if config.tracking.enabled:
            mlflow.log_metrics({f"trial.{key}": value for key, value in metric_map.items()})
            log_artifacts(
                [
                    artifact_dir / ARTIFACT_MANIFEST_FILENAME,
                    artifact_dir / MODEL_STATE_FILENAME,
                    report_path,
                ]
            )
        return metric_value
    finally:
        if run_context is not None:
            run_context.__exit__(None, None, None)


def run(config: ExperimentConfig) -> None:
    if config.tracking.enabled:
        configure_mlflow(config)

    run_context = start_run_if_enabled(
        config,
        run_name=f"study-{config.chain.name.value}-{config.model.family.value}-{config.max_delay_seconds}s",
    )
    try:
        if run_context is not None:
            run_context.__enter__()
            log_config(config)
            mlflow.set_tags(config.tracking.tags)

        study = optuna.create_study(
            study_name=config.tuning.study_name,
            direction=config.tuning.direction,
            pruner=(
                optuna.pruners.MedianPruner()
                if config.tuning.prune
                else optuna.pruners.NopPruner()
            ),
            sampler=optuna.samplers.TPESampler(seed=config.tuning.sampler_seed),
        )
        study.optimize(
            lambda trial: _objective(config, trial),
            n_trials=config.tuning.n_trials,
            timeout=config.tuning.timeout_seconds,
        )
        tuning_root = Path(config.paths.tuning_root)
        study_path = tuning_root / "study.json"
        trials_path = tuning_root / "trials.json"
        _write_json(study_path, _study_summary(config, study))
        _write_json(trials_path, [_trial_record(trial) for trial in study.trials])
        if config.tracking.enabled:
            metrics = {"study.n_trials": float(len(study.trials))}
            completed_trials = [
                trial for trial in study.trials if trial.state == TrialState.COMPLETE
            ]
            if completed_trials:
                metrics["study.best_value"] = study.best_value
                mlflow.log_params(
                    {
                        f"study.best_param.{key}": str(value)
                        for key, value in study.best_params.items()
                    }
                )
            mlflow.log_metrics(metrics)
            log_artifacts([study_path, trials_path])
    finally:
        if run_context is not None:
            run_context.__exit__(None, None, None)


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="tune"))


if __name__ == "__main__":
    main()

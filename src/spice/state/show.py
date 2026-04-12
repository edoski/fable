"""Inspection helpers for `spice show`."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import optuna

from . import (
    ARTIFACT_ROOT_KIND,
    DATASET_ROOT_KIND,
    STUDY_ROOT_KIND,
    detect_root_kind,
    state_db_path,
)
from .artifact import (
    list_simulation_runs,
    list_training_epochs,
    load_artifact_manifest,
    load_simulation_summary,
    load_training_summary,
)
from .dataset import list_acquire_runs, load_dataset_summary
from .engine import table_exists
from .schema import artifact_manifest
from .study import load_study, study_storage


def describe_root(root: Path, *, detail: str | None = None) -> dict[str, object]:
    db_path = state_db_path(root)
    root_kind = detect_root_kind(db_path)
    payload: dict[str, object] = {
        "root": str(root),
        "state_db": str(db_path),
        "root_kind": root_kind,
    }
    if root_kind == DATASET_ROOT_KIND:
        payload["dataset"] = _dataset_payload(load_dataset_summary(db_path))
        if detail == "runs":
            payload["runs"] = list_acquire_runs(db_path)
        return payload
    if root_kind == ARTIFACT_ROOT_KIND:
        payload.update(_artifact_payload(db_path, detail=detail))
        return payload
    if root_kind == STUDY_ROOT_KIND:
        payload.update(_study_payload(db_path, detail=detail))
        if table_exists(db_path, artifact_manifest.name):
            payload.update(_artifact_payload(db_path, detail=detail))
        return payload
    raise ValueError(f"Unsupported root kind: {root_kind}")


def sectioned_summary(
    payload: dict[str, object],
) -> tuple[str, list[tuple[str, list[tuple[str, str]]]]]:
    root_kind = payload["root_kind"]
    if root_kind == DATASET_ROOT_KIND:
        dataset = _mapping(payload["dataset"])
        sections = [
            (
                "dataset",
                [
                    ("id", str(dataset["dataset_id"])),
                    ("chain", str(dataset["chain"])),
                    ("provider", str(dataset["provider"])),
                    ("state", str(payload["state_db"])),
                ],
            ),
            (
                "request",
                [
                    ("history", _window_string(_mapping(dataset["history_request"]))),
                    ("evaluation", _window_string(_mapping(dataset["evaluation_request"]))),
                ],
            ),
            (
                "coverage",
                [
                    ("history rows", str(dataset["history_rows"])),
                    ("evaluation rows", str(dataset["evaluation_rows"])),
                ],
            ),
        ]
        runs = payload.get("runs")
        if isinstance(runs, list):
            sections.append(
                (
                    "runs",
                    [
                        (f"run {index}", _acquire_run_string(_mapping(run)))
                        for index, run in enumerate(runs, start=1)
                    ],
                )
            )
        return "dataset summary", sections
    if root_kind == ARTIFACT_ROOT_KIND:
        return "artifact summary", _artifact_sections(payload)
    if root_kind == STUDY_ROOT_KIND:
        sections = _study_sections(payload)
        if "manifest" in payload:
            sections.extend(_artifact_sections(payload))
        return "study summary", sections
    raise ValueError(f"Unsupported root kind: {root_kind}")


def _artifact_payload(db_path: Path, *, detail: str | None) -> dict[str, object]:
    payload: dict[str, object] = {}
    manifest = load_artifact_manifest(db_path)
    training = load_training_summary(db_path)
    simulation = load_simulation_summary(db_path)
    payload["manifest"] = {
        "chain": manifest.chain.name,
        "dataset_id": manifest.dataset_id,
        "task_id": manifest.task_id,
        "variant": manifest.variant.value,
        "study_id": None if manifest.study is None else manifest.study.id,
        "model_id": manifest.model.id,
        "max_supported_delay_seconds": manifest.max_supported_delay_seconds,
        "lookback_seconds": manifest.lookback_seconds,
        "sample_count": manifest.sample_count,
        "feature_set_id": manifest.feature_set_id,
        "feature_names": list(manifest.feature_names),
    }
    if training is not None:
        payload["training"] = _training_summary_payload(training)
    if simulation is not None:
        payload["simulation"] = _simulation_summary_payload(simulation)
    if detail == "epochs":
        payload["epochs"] = list_training_epochs(db_path)
    if detail == "runs":
        payload["runs"] = [asdict(run) for run in list_simulation_runs(db_path)]
    return payload


def _study_payload(db_path: Path, *, detail: str | None) -> dict[str, object]:
    summaries = optuna.get_all_study_summaries(storage=study_storage(db_path))
    if not summaries:
        raise ValueError(f"No Optuna study found in {db_path}")
    if len(summaries) != 1:
        raise ValueError(f"Expected exactly one study in {db_path}, found {len(summaries)}")
    study_name = summaries[0].study_name
    study = load_study(db_path, study_name=study_name)
    context = study.user_attrs.get("spice_context")
    if not isinstance(context, dict):
        raise ValueError(f"Missing SPICE study context in {db_path}")
    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_trial_number = best_trial.number
    except ValueError:
        best_value = None
        best_trial_number = None
    payload: dict[str, object] = {
        "study": {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "best_value": best_value,
            "best_trial_number": best_trial_number,
            "trial_count": len(study.trials),
            "context": dict(context),
        }
    }
    if detail == "trials":
        payload["trials"] = [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "best_epoch": trial.user_attrs.get("spice_best_epoch"),
                "params": trial.user_attrs.get("spice_params"),
            }
            for trial in study.trials
        ]
    return payload


def _artifact_sections(payload: dict[str, object]) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = _mapping(payload["manifest"])
    sections = [
        (
            "artifact",
            [
                ("chain", str(manifest["chain"])),
                ("dataset", str(manifest["dataset_id"])),
                ("task", str(manifest["task_id"])),
                ("model", str(manifest["model_id"])),
                ("variant", str(manifest["variant"])),
                ("study", str(manifest["study_id"] or "n/a")),
                ("capability", f"{manifest['max_supported_delay_seconds']}s"),
            ],
        ),
    ]
    training = payload.get("training")
    if isinstance(training, dict):
        sections.append(
            (
                "training",
                [
                    ("best epoch", str(training["best_epoch"])),
                    ("split", str(training["split"])),
                    (
                        "validation loss",
                        _metric_value(training, "best_validation_metrics", "total_loss"),
                    ),
                    (
                        "validation accuracy",
                        _metric_value(training, "best_validation_metrics", "accuracy"),
                    ),
                    (
                        "test profit over baseline",
                        _metric_value(training, "test_metrics", "mean_profit_over_baseline"),
                    ),
                ],
            )
        )
    simulation = payload.get("simulation")
    if isinstance(simulation, dict):
        sections.append(
            (
                "simulation",
                [
                    ("requested", f"{simulation['requested_delay_seconds']}s"),
                    ("window", f"{simulation['simulation_window_seconds']}s"),
                    ("repetitions", str(simulation["repetitions"])),
                    ("events", str(simulation["total_events"])),
                    ("profit mean", _aggregate_value(simulation, "profit_over_baseline", "mean")),
                    ("cost mean", _aggregate_value(simulation, "cost_over_optimum", "mean")),
                ],
            )
        )
    epochs = payload.get("epochs")
    if isinstance(epochs, list):
        sections.append(
            (
                "epochs",
                [(f"epoch {row['epoch']}", _epoch_string(_mapping(row))) for row in epochs],
            )
        )
    runs = payload.get("runs")
    if isinstance(runs, list):
        sections.append(
            (
                "runs",
                [
                    (f"run {index}", _simulation_run_string(_mapping(row)))
                    for index, row in enumerate(runs, start=1)
                ],
            )
        )
    return sections


def _study_sections(payload: dict[str, object]) -> list[tuple[str, list[tuple[str, str]]]]:
    study = _mapping(payload["study"])
    context = _mapping(study["context"])
    sections = [
        (
            "study",
            [
                ("id", str(study["study_name"])),
                ("direction", str(study["direction"])),
                ("chain", str(context["chain"])),
                ("dataset", str(context["dataset_id"])),
                ("task", str(context["task_id"])),
                ("feature set", str(context["feature_set_id"])),
                ("model", str(context["model_id"])),
                ("trials", str(study["trial_count"])),
                ("state", str(payload["state_db"])),
            ],
        ),
        (
            "best trial",
            [
                (
                    "number",
                    "n/a"
                    if study["best_trial_number"] is None
                    else str(study["best_trial_number"]),
                ),
                (
                    "value",
                    "n/a"
                    if study["best_value"] is None
                    else f"{float(study['best_value']):.4f}",
                ),
            ],
        ),
    ]
    trials = payload.get("trials")
    if isinstance(trials, list):
        sections.append(
            (
                "trials",
                [(f"trial {row['number']}", _trial_string(_mapping(row))) for row in trials],
            )
        )
    return sections


def _dataset_payload(summary) -> dict[str, object]:
    return {
        "dataset_id": summary.dataset.id,
        "chain": summary.chain.name,
        "provider": summary.provider.name,
        "history_request": {
            "start_timestamp": summary.request.history.start_timestamp,
            "end_timestamp": summary.request.history.end_timestamp,
        },
        "evaluation_request": {
            "start_timestamp": summary.request.evaluation.start_timestamp,
            "end_timestamp": summary.request.evaluation.end_timestamp,
        },
        "history_rows": summary.validation.history.rows,
        "evaluation_rows": summary.validation.evaluation.rows,
    }


def _training_summary_payload(summary) -> dict[str, object]:
    return {
        "best_epoch": summary.best_epoch,
        "split": (
            f"train={summary.split_sizes.train_samples} "
            f"validation={summary.split_sizes.validation_samples} "
            f"test={summary.split_sizes.test_samples}"
        ),
        "best_validation_metrics": asdict(summary.best_validation_metrics),
        "test_metrics": asdict(summary.test_metrics),
    }


def _simulation_summary_payload(summary) -> dict[str, object]:
    return {
        "requested_delay_seconds": summary.requested_delay_seconds,
        "simulation_window_seconds": summary.simulation_window_seconds,
        "repetitions": summary.repetitions,
        "total_events": summary.total_events,
        "profit_over_baseline": asdict(summary.profit_over_baseline),
        "cost_over_optimum": asdict(summary.cost_over_optimum),
    }


def _window_string(window: dict[str, object]) -> str:
    return f"{window['start_timestamp']} -> {window['end_timestamp']}"


def _acquire_run_string(run: dict[str, object]) -> str:
    return (
        f"task={run['task_id']} feature_set={run['feature_set_id']} "
        f"history={run['required_history_blocks']} blocks"
    )


def _metric_value(summary: dict[str, object], metric_group: str, metric_name: str) -> str:
    metrics = _mapping(summary[metric_group])
    return f"{float(metrics[metric_name]):.4f}"


def _aggregate_value(summary: dict[str, object], metric_group: str, metric_name: str) -> str:
    aggregate = _mapping(summary[metric_group])
    return f"{float(aggregate[metric_name]):.4f}"


def _epoch_string(row: dict[str, object]) -> str:
    train = _mapping(row["train_metrics"])
    validation = _mapping(row["validation_metrics"])
    return (
        f"train_loss={float(train['total_loss']):.4f} "
        f"val_loss={float(validation['total_loss']):.4f}"
    )


def _simulation_run_string(row: dict[str, object]) -> str:
    return (
        f"events={int(row['n_events'])} "
        f"profit={float(row['profit_over_baseline']):.4f} "
        f"cost={float(row['cost_over_optimum']):.4f}"
    )


def _trial_string(row: dict[str, object]) -> str:
    value = "n/a" if row["value"] is None else f"{float(row['value']):.4f}"
    return f"state={row['state']} value={value}"


def _mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Expected mapping payload")
    return dict(payload)

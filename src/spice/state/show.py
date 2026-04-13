"""Inspection helpers for selector-based `spice show` commands."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

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
from .study import describe_study


def describe_root(root: Path, *, detail: str | None = None) -> dict[str, object]:
    db_path = state_db_path(root)
    root_kind = detect_root_kind(db_path)
    if root_kind == DATASET_ROOT_KIND:
        payload: dict[str, object] = {
            "root_kind": root_kind,
            "dataset": _dataset_payload(load_dataset_summary(db_path)),
        }
        if detail == "runs":
            payload["runs"] = list_acquire_runs(db_path)
        return payload
    if root_kind == ARTIFACT_ROOT_KIND:
        return {
            "root_kind": root_kind,
            **_artifact_payload(db_path, detail=detail),
        }
    if root_kind == STUDY_ROOT_KIND:
        return {
            "root_kind": root_kind,
            **describe_study(db_path, detail=detail),
        }
    raise ValueError(f"Unsupported root kind: {root_kind}")


def sectioned_summary(
    payload: dict[str, object],
) -> tuple[str, list[tuple[str, list[tuple[str, str]]]]]:
    root_kind = str(payload["root_kind"])
    if root_kind == DATASET_ROOT_KIND:
        return "dataset summary", _dataset_sections(payload)
    if root_kind == ARTIFACT_ROOT_KIND:
        return "artifact summary", _artifact_sections(payload)
    if root_kind == STUDY_ROOT_KIND:
        return "study summary", _study_sections(payload)
    raise ValueError(f"Unsupported root kind: {root_kind}")


def _dataset_payload(summary) -> dict[str, object]:
    return {
        "dataset_id": summary.dataset.id,
        "dataset_name": summary.dataset.name,
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
        "history_coverage": {
            "start_timestamp": summary.coverage.history.start_timestamp,
            "end_timestamp": summary.coverage.history.end_timestamp,
        },
        "evaluation_coverage": {
            "start_timestamp": summary.coverage.evaluation.start_timestamp,
            "end_timestamp": summary.coverage.evaluation.end_timestamp,
        },
        "history_rows": summary.validation.history.rows,
        "evaluation_rows": summary.validation.evaluation.rows,
    }


def _artifact_payload(db_path: Path, *, detail: str | None) -> dict[str, object]:
    manifest = load_artifact_manifest(db_path)
    training = load_training_summary(db_path)
    simulation = load_simulation_summary(db_path)
    payload: dict[str, object] = {
        "manifest": {
            "artifact_id": manifest.artifact_id,
            "chain": manifest.chain.name,
            "dataset_id": manifest.dataset_id,
            "dataset_name": manifest.dataset_name,
            "task_id": manifest.task_id,
            "variant": manifest.variant.value,
            "study_id": manifest.study_id,
            "study_name": None if manifest.study is None else manifest.study.name,
            "model_id": manifest.model.id,
            "max_supported_delay_seconds": manifest.max_supported_delay_seconds,
            "lookback_seconds": manifest.lookback_seconds,
            "feature_history_seconds": manifest.feature_history_seconds,
            "sample_count": manifest.sample_count,
            "max_candidate_slots": manifest.max_candidate_slots,
            "feature_set_id": manifest.feature_set_id,
            "feature_names": list(manifest.feature_names),
        }
    }
    if training is not None:
        payload["training"] = {
            "best_epoch": training.best_epoch,
            "split": (
                f"train={training.split_sizes.train_samples} "
                f"validation={training.split_sizes.validation_samples} "
                f"test={training.split_sizes.test_samples}"
            ),
            "best_validation_metrics": asdict(training.best_validation_metrics),
            "test_metrics": asdict(training.test_metrics),
        }
    if simulation is not None:
        payload["simulation"] = {
            "requested_delay_seconds": simulation.requested_delay_seconds,
            "simulation_window_seconds": simulation.simulation_window_seconds,
            "repetitions": simulation.repetitions,
            "total_events": simulation.total_events,
            "profit_over_baseline": asdict(simulation.profit_over_baseline),
            "cost_over_optimum": asdict(simulation.cost_over_optimum),
        }
    if detail == "epochs":
        payload["epochs"] = list_training_epochs(db_path)
    if detail == "runs":
        payload["runs"] = [asdict(run) for run in list_simulation_runs(db_path)]
    return payload


def _dataset_sections(payload: dict[str, object]) -> list[tuple[str, list[tuple[str, str]]]]:
    dataset = _mapping(payload["dataset"])
    sections = [
        (
            "dataset",
            [
                ("name", str(dataset["dataset_name"])),
                ("storage id", str(dataset["dataset_id"])),
                ("chain", str(dataset["chain"])),
                ("provider", str(dataset["provider"])),
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
                ("history", _window_string(_mapping(dataset["history_coverage"]))),
                ("evaluation", _window_string(_mapping(dataset["evaluation_coverage"]))),
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
    return sections


def _artifact_sections(payload: dict[str, object]) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = _mapping(payload["manifest"])
    sections = [
        (
            "artifact",
            [
                ("artifact id", str(manifest["artifact_id"])),
                ("dataset", str(manifest["dataset_name"])),
                ("dataset id", str(manifest["dataset_id"])),
                ("chain", str(manifest["chain"])),
                ("task", str(manifest["task_id"])),
                ("feature set", str(manifest["feature_set_id"])),
                ("model", str(manifest["model_id"])),
                ("variant", str(manifest["variant"])),
                ("study", str(manifest["study_name"] or "n/a")),
                ("study id", str(manifest["study_id"] or "n/a")),
                ("capability", f"{manifest['max_supported_delay_seconds']}s"),
                ("lookback", f"{manifest['lookback_seconds']}s"),
                ("feature history", f"{manifest['feature_history_seconds']}s"),
                ("max slots", str(manifest["max_candidate_slots"])),
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
    sections = [
        (
            "study",
            [
                ("name", str(study["study_name"])),
                ("storage id", str(study["study_id"])),
                ("direction", str(study["direction"])),
                ("objective", str(study["objective_metric"])),
                ("chain", str(study["chain_name"])),
                ("dataset", str(study["dataset_name"])),
                ("dataset id", str(study["dataset_id"])),
                ("task", str(study["task_id"])),
                ("feature set", str(study["feature_set_id"])),
                ("model", str(study["model_id"])),
                (
                    "sampler",
                    f"{study['sampler_name']} seed={study['sampler_seed']}",
                ),
                ("pruner", str(study["pruner_name"])),
                ("trials", str(study["trial_count"])),
            ],
        ),
        (
            "best trial",
            [
                (
                    "number",
                    "n/a"
                    if study["best_trial_number"] is None
                    else str(study["best_trial_number"] + 1),
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
    config = payload.get("config")
    if isinstance(config, dict):
        config_mapping = _mapping(config)
        sections.extend(
            [
                ("task config", _mapping_fields(_mapping(config_mapping["task"]))),
                ("feature set config", _mapping_fields(_mapping(config_mapping["feature_set"]))),
                ("model config", _mapping_fields(_mapping(config_mapping["model"]))),
                ("split config", _mapping_fields(_mapping(config_mapping["split"]))),
                ("training config", _mapping_fields(_mapping(config_mapping["training"]))),
                ("tuning config", _mapping_fields(_mapping(config_mapping["tuning"]))),
                ("tuning space", _mapping_fields(_mapping(config_mapping["tuning_space"]))),
            ]
        )
    trials = payload.get("trials")
    if isinstance(trials, list):
        sections.append(
            (
                "trials",
                [(f"trial {row['number'] + 1}", _trial_string(_mapping(row))) for row in trials],
            )
        )
    return sections


def _window_string(window: dict[str, object]) -> str:
    return f"{window['start_timestamp']} -> {window['end_timestamp']}"


def _acquire_run_string(run: dict[str, object]) -> str:
    return (
        f"task={run['task_id']} feature_set={run['feature_set_id']} "
        f"history={run['acquired_history_window_seconds']}s "
        f"anchors={run['valid_anchor_samples']}"
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
    best_epoch = row.get("best_epoch")
    params = row.get("params")
    params_text = "" if not isinstance(params, dict) or not params else f" params={params}"
    epoch_text = "" if best_epoch is None else f" best_epoch={best_epoch}"
    return f"state={row['state']} value={value}{epoch_text}{params_text}"


def _mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Expected mapping payload")
    return dict(payload)


def _mapping_fields(payload: dict[str, object]) -> list[tuple[str, str]]:
    return [
        (str(key).replace("_", " "), _value_string(value))
        for key, value in payload.items()
    ]


def _value_string(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)

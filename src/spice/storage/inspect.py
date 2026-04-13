"""Inspection helpers for selector-based `spice show` commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..corpus.metadata import AcquireRunRecord, DatasetSummary
from ..features import FeaturePrerequisites
from ..modeling.artifacts import TrainingArtifactManifest
from ..modeling.results import SimulationSummaryRecord, TrainingEpochRecord, TrainingSummary
from ..modeling.simulation import SimulationRunSummary
from ..prediction import MetricDescriptor
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
from .catalog import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from .corpus import list_acquire_runs, load_dataset_summary
from .study import StudySummary, StudyTrialRecord, list_trial_records, load_study_summary


@dataclass(frozen=True, slots=True)
class DatasetRootDescription:
    summary: DatasetSummary
    runs: list[AcquireRunRecord] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRootDescription:
    manifest: TrainingArtifactManifest
    training: TrainingSummary | None = None
    simulation: SimulationSummaryRecord | None = None
    epochs: list[TrainingEpochRecord] | None = None
    runs: list[SimulationRunSummary] | None = None


@dataclass(frozen=True, slots=True)
class StudyRootDescription:
    summary: StudySummary
    include_config: bool = False
    trials: list[StudyTrialRecord] | None = None


RootDescription = DatasetRootDescription | ArtifactRootDescription | StudyRootDescription


def dataset_list_sections(
    records: list[CatalogDatasetRecord],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "datasets",
            [
                (
                    record.dataset_name,
                    (
                        f"chain={record.chain_name} "
                        f"provider={record.provider_name} "
                        f"id={record.dataset_id}"
                    ),
                )
                for record in records
            ],
        )
    ]


def study_list_sections(
    records: list[CatalogStudyRecord],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "studies",
            [
                (
                    record.study_name,
                    (
                        f"chain={record.chain_name} "
                        f"dataset={record.dataset_name} "
                        f"feature_set={record.feature_set_id} "
                        f"prediction={record.prediction_id} "
                        f"model={record.model_id} "
                        f"problem={record.problem_id} "
                        f"id={record.study_id}"
                    ),
                )
                for record in records
            ],
        )
    ]


def artifact_list_sections(
    records: list[CatalogArtifactRecord],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "artifacts",
            [
                (
                    record.artifact_id,
                    (
                        f"chain={record.chain_name} "
                        f"dataset={record.dataset_name} "
                        f"feature_set={record.feature_set_id} "
                        f"prediction={record.prediction_id} "
                        f"model={record.model_id} "
                        f"problem={record.problem_id} "
                        f"variant={record.variant}"
                        + (
                            ""
                            if record.study_name is None
                            else f" study={record.study_name}"
                        )
                    ),
                )
                for record in records
            ],
        )
    ]


def describe_root(root: Path, *, detail: str | None = None) -> RootDescription:
    db_path = state_db_path(root)
    root_kind = detect_root_kind(db_path)
    if root_kind == DATASET_ROOT_KIND:
        return DatasetRootDescription(
            summary=load_dataset_summary(db_path),
            runs=list_acquire_runs(db_path) if detail == "runs" else None,
        )
    if root_kind == ARTIFACT_ROOT_KIND:
        return ArtifactRootDescription(
            manifest=load_artifact_manifest(db_path),
            training=load_training_summary(db_path),
            simulation=load_simulation_summary(db_path),
            epochs=list_training_epochs(db_path) if detail == "epochs" else None,
            runs=list_simulation_runs(db_path) if detail == "runs" else None,
        )
    if root_kind == STUDY_ROOT_KIND:
        return StudyRootDescription(
            summary=load_study_summary(db_path),
            include_config=detail == "config",
            trials=list_trial_records(db_path) if detail == "trials" else None,
        )
    raise ValueError(f"Unsupported root kind: {root_kind}")


def sectioned_summary(
    description: RootDescription,
) -> tuple[str, list[tuple[str, list[tuple[str, str]]]]]:
    if isinstance(description, DatasetRootDescription):
        return "dataset summary", _dataset_sections(description)
    if isinstance(description, ArtifactRootDescription):
        return "artifact summary", _artifact_sections(description)
    if isinstance(description, StudyRootDescription):
        return "study summary", _study_sections(description)
    raise ValueError(f"Unsupported root description: {type(description).__name__}")


def _dataset_sections(
    description: DatasetRootDescription,
) -> list[tuple[str, list[tuple[str, str]]]]:
    summary = description.summary
    sections = [
        (
            "dataset",
            [
                ("name", summary.dataset.name),
                ("storage id", summary.dataset.id),
                ("chain", summary.chain.name),
                ("provider", summary.provider.name),
            ],
        ),
        (
            "request",
            [
                ("history", _window_string(summary.request.history)),
                ("evaluation", _window_string(summary.request.evaluation)),
            ],
        ),
        (
            "coverage",
            [
                ("history", _window_string(summary.coverage.history)),
                ("evaluation", _window_string(summary.coverage.evaluation)),
                ("history rows", str(summary.validation.history.rows)),
                ("evaluation rows", str(summary.validation.evaluation.rows)),
            ],
        ),
    ]
    if description.runs:
        sections.append(
            (
                "runs",
                [
                    (f"run {index}", _acquire_run_string(run))
                    for index, run in enumerate(description.runs, start=1)
                ],
            )
        )
    return sections


def _artifact_sections(
    description: ArtifactRootDescription,
) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = description.manifest
    sections = [
        (
            "artifact",
            [
                ("artifact id", manifest.artifact_id),
                ("objective", manifest.objective_id),
                ("prediction", manifest.prediction_id),
                ("dataset", manifest.dataset_name),
                ("dataset id", manifest.dataset_id),
                ("chain", manifest.chain.name),
                ("problem", manifest.problem_id),
                ("feature set", manifest.feature_set_id),
                ("feature family", manifest.feature_family_id),
                ("model", manifest.model.id),
                ("variant", manifest.variant.value),
                ("study", manifest.study.name if manifest.study is not None else "n/a"),
                ("study id", manifest.study_id or "n/a"),
                ("capability", f"{manifest.max_supported_delay_seconds}s"),
                ("lookback", f"{manifest.lookback_seconds}s"),
                (
                    "feature prerequisites",
                    _feature_prerequisites_string(manifest.feature_prerequisites),
                ),
                ("max slots", str(manifest.max_candidate_slots)),
            ],
        ),
    ]
    if description.training is not None:
        training = description.training
        sections.append(
            (
                "training",
                [
                    ("best epoch", str(training.best_epoch)),
                    ("representation", training.representation_id),
                    ("storage mode", training.storage_mode_id),
                    ("batch planner", training.batch_planner_id),
                    ("family execution", training.family_execution_id),
                    (
                        "split",
                        " ".join(
                            [
                                f"train={training.split_sizes.train_samples}",
                                f"validation={training.split_sizes.validation_samples}",
                                f"test={training.split_sizes.test_samples}",
                            ]
                        ),
                    ),
                    (
                        "validation metrics",
                        _metric_bundle_string(
                            training.metric_descriptors,
                            training.best_validation_metrics.values,
                        ),
                    ),
                    (
                        "test metrics",
                        _metric_bundle_string(
                            training.metric_descriptors,
                            training.test_metrics.values,
                        ),
                    ),
                ],
            )
        )
    if description.simulation is not None:
        simulation = description.simulation
        sections.append(
            (
                "simulation",
                [
                    ("requested", f"{simulation.requested_delay_seconds}s"),
                    ("window", f"{simulation.simulation_window_seconds}s"),
                    ("repetitions", str(simulation.repetitions)),
                    ("events", str(simulation.total_events)),
                    (
                        "metrics",
                        _metric_bundle_string(
                            simulation.metric_descriptors,
                            simulation.metrics.values,
                        ),
                    ),
                ],
            )
        )
    if description.epochs:
        sections.append(
            (
                "epochs",
                [
                    (f"epoch {record.epoch}", _epoch_string(record))
                    for record in description.epochs
                ],
            )
        )
    if description.runs:
        sections.append(
            (
                "runs",
                [
                    (f"run {index}", _simulation_run_string(run))
                    for index, run in enumerate(description.runs, start=1)
                ],
            )
        )
    return sections


def _study_sections(
    description: StudyRootDescription,
) -> list[tuple[str, list[tuple[str, str]]]]:
    summary = description.summary
    manifest = summary.manifest
    best_trial = summary.best_trial
    sections = [
        (
            "study",
            [
                ("name", manifest.study_name),
                ("storage id", manifest.study_id),
                ("objective", manifest.objective_id),
                ("prediction", manifest.prediction_id),
                ("chain", manifest.chain_name),
                ("dataset", manifest.dataset_name),
                ("dataset id", manifest.dataset_id),
                ("problem", manifest.problem_id),
                ("feature set", manifest.feature_set_id),
                ("model", manifest.model_id),
                ("sampler", f"{manifest.sampler_name} seed={manifest.sampler_seed}"),
                ("pruner", manifest.pruner_name),
                ("trials", str(summary.trial_counts.total)),
            ],
        ),
        (
            "best trial",
            [
                (
                    "number",
                    "n/a" if best_trial is None else str(best_trial.number + 1),
                ),
                (
                    "value",
                    (
                        "n/a"
                        if best_trial is None or best_trial.value is None
                        else _metric_string(best_trial.value)
                    ),
                ),
            ],
        ),
    ]
    if description.include_config:
        sections.extend(
            [
                ("problem config", _mapping_fields(manifest.problem.model_dump(mode="json"))),
                (
                    "feature set config",
                    _mapping_fields(manifest.feature_set.model_dump(mode="json")),
                ),
                (
                    "prediction config",
                    _mapping_fields(manifest.prediction.model_dump(mode="json")),
                ),
                (
                    "model config",
                    _mapping_fields(manifest.model.model_dump(mode="json", exclude_none=True)),
                ),
                ("split config", _mapping_fields(manifest.split.model_dump(mode="json"))),
                (
                    "training config",
                    _mapping_fields(manifest.training.model_dump(mode="json")),
                ),
                (
                    "tuning config",
                    _mapping_fields(
                        {
                            "sampler_name": manifest.sampler_name,
                            "sampler_seed": manifest.sampler_seed,
                            "pruner_name": manifest.pruner_name,
                            "enable_pruning": manifest.enable_pruning,
                        }
                    ),
                ),
                (
                    "tuning space",
                    _mapping_fields(
                        manifest.tuning_space.model_dump(mode="json", exclude_none=True)
                    ),
                ),
            ]
        )
    if description.trials:
        sections.append(
            (
                "trials",
                [
                    (f"trial {record.number + 1}", _trial_string(record))
                    for record in description.trials
                ],
            )
        )
    return sections


def _window_string(window) -> str:
    return f"{window.start_timestamp} -> {window.end_timestamp}"


def _acquire_run_string(run: AcquireRunRecord) -> str:
    return (
        f"problem={run.problem.problem_id} feature_set={run.problem.feature_set_id} "
        f"family={run.problem.feature_family_id} "
        f"history={run.problem.acquired_history_window_seconds}s "
        f"anchors={run.problem.valid_anchor_samples}"
    )


def _feature_prerequisites_string(prerequisites: FeaturePrerequisites) -> str:
    return (
        f"history={prerequisites.history_seconds}s "
        f"warmup={prerequisites.warmup_rows} rows"
    )


def _metric_string(value: float) -> str:
    return f"{value:.4f}"


def _epoch_string(record: TrainingEpochRecord) -> str:
    return (
        f"train={_metric_bundle_string([], record.train_metrics.values)} "
        f"val={_metric_bundle_string([], record.validation_metrics.values)}"
    )


def _simulation_run_string(run: SimulationRunSummary) -> str:
    return (
        f"events={run.n_events} "
        f"metrics={_metric_bundle_string([], run.metrics)}"
    )


def _trial_string(record: StudyTrialRecord) -> str:
    value = "n/a" if record.value is None else f"{record.value:.4f}"
    params = record.params.model_dump(mode="json", exclude_none=True)
    params_text = "" if not params else f" params={params}"
    epoch_text = "" if record.best_epoch is None else f" best_epoch={record.best_epoch}"
    return f"state={record.state.value} value={value}{epoch_text}{params_text}"


def _mapping_fields(payload: dict[str, object]) -> list[tuple[str, str]]:
    return [
        (str(key).replace("_", " "), _value_string(value))
        for key, value in payload.items()
    ]


def _metric_bundle_string(
    descriptors: list[MetricDescriptor],
    metrics: dict[str, float],
) -> str:
    if descriptors:
        ordered = [
            f"{descriptor.id}={metrics[descriptor.id]:.4f}"
            for descriptor in descriptors
            if descriptor.id in metrics
        ]
        if ordered:
            return " ".join(ordered)
    return " ".join(f"{metric_id}={value:.4f}" for metric_id, value in sorted(metrics.items()))


def _value_string(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)

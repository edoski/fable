"""Artifact-root inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.rendering import metric_bundle_string, window_metric_fields
from ..features import FeaturePrerequisites
from ..modeling.artifacts import TrainingArtifactManifest
from ..modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary, TrainingEpochRecord
from .artifact import (
    list_evaluation_summaries,
    list_training_epochs,
    load_artifact_manifest,
    load_training_summary,
)
from .catalog import CatalogArtifactRecord


@dataclass(frozen=True, slots=True)
class ArtifactRootDescription:
    manifest: TrainingArtifactManifest
    training: LoadedTrainingSummary | None = None
    evaluations: list[LoadedEvaluationSummary] | None = None
    epochs: list[TrainingEpochRecord] | None = None
    show_runs: bool = False


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
                        f"features={record.features_id} "
                        f"prediction={record.prediction_id} "
                        f"model={record.model_id} "
                        f"problem={record.problem_id} "
                        f"variant={record.variant}"
                        + ("" if record.study_name is None else f" study={record.study_name}")
                    ),
                )
                for record in records
            ],
        )
    ]


def describe_artifact_root(
    root_db_path: Path,
    *,
    detail: str | None = None,
) -> ArtifactRootDescription:
    training = load_training_summary(root_db_path)
    evaluations = list_evaluation_summaries(root_db_path)
    return ArtifactRootDescription(
        manifest=(
            training.manifest
            if training is not None
            else evaluations[0].manifest
            if evaluations
            else load_artifact_manifest(root_db_path)
        ),
        training=training,
        evaluations=evaluations or None,
        epochs=list_training_epochs(root_db_path) if detail == "epochs" else None,
        show_runs=detail == "runs",
    )


def artifact_sections(
    description: ArtifactRootDescription,
) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = description.manifest
    sections = [
        (
            "artifact",
            [
                ("artifact id", manifest.artifact_id),
                ("prediction", manifest.prediction_id),
                ("dataset", manifest.dataset_name),
                ("dataset id", manifest.dataset_id),
                ("chain", manifest.chain_name),
                ("problem", manifest.problem_id),
                ("execution policy", manifest.semantics.execution_policy.execution_policy_id),
                ("dataset builder", manifest.dataset_builder_id),
                ("features", manifest.features_id),
                ("model", manifest.model.id),
                ("variant", manifest.variant.value),
                ("study", manifest.study.name if manifest.study is not None else "n/a"),
                ("study id", manifest.study_id or "n/a"),
                ("capability", f"{manifest.max_delay_seconds}s"),
                ("action width", str(manifest.action_width)),
                ("lookback", f"{manifest.lookback_seconds}s"),
                (
                    "feature prerequisites",
                    feature_prerequisites_string(manifest.feature_prerequisites),
                ),
            ],
        ),
    ]
    if description.training is not None:
        training = description.training
        runtime = training.runtime
        sections.append(
            (
                "training",
                [
                    ("best epoch", str(runtime.best_epoch)),
                    (
                        "objective",
                        (
                            f"{training.manifest.semantics.objective.objective_id}:"
                            f"{runtime.best_objective_metric_id}"
                        ),
                    ),
                    ("best objective", f"{runtime.best_objective_value:.4f}"),
                    (
                        "rows",
                        f"used={runtime.n_rows_used} available={runtime.n_rows_available}",
                    ),
                    (
                        "split",
                        " ".join(
                            [
                                f"train={runtime.split_sizes.train_samples}",
                                f"validation={runtime.split_sizes.validation_samples}",
                                f"test={runtime.split_sizes.test_samples}",
                            ]
                        ),
                    ),
                    (
                        "validation metrics",
                        metric_bundle_string(
                            training.manifest.training_metric_descriptors,
                            runtime.best_validation_metrics.values,
                        ),
                    ),
                    (
                        "test metrics",
                        metric_bundle_string(
                            training.manifest.training_metric_descriptors,
                            runtime.test_metrics.values,
                        ),
                    ),
                ],
            )
        )
    if description.evaluations:
        multiple_evaluations = len(description.evaluations) > 1
        for evaluation in description.evaluations:
            runtime = evaluation.runtime
            evaluation_title = (
                "evaluation"
                if not multiple_evaluations
                else f"evaluation {runtime.evaluation_id} {runtime.delay_seconds}s"
            )
            sections.append(
                (
                    evaluation_title,
                    [
                        ("evaluation id", evaluation.evaluation_id),
                        ("requested", f"{runtime.delay_seconds}s"),
                        ("evaluation", runtime.evaluation_id),
                        (
                            "execution policy",
                            evaluation.manifest.semantics.execution_policy.execution_policy_id,
                        ),
                        ("events", str(runtime.total_events)),
                        (
                            "metrics",
                            metric_bundle_string(
                                runtime.metric_descriptors,
                                runtime.metrics.values,
                            ),
                        ),
                    ],
                )
            )
            if runtime.window_metrics:
                sections.append(
                    (
                        (
                            "evaluation windows"
                            if not multiple_evaluations
                            else f"{evaluation_title} windows"
                        ),
                        window_metric_fields(
                            runtime.metric_descriptors,
                            runtime.window_metrics,
                        ),
                    )
                )
    if description.epochs:
        sections.append(
            (
                "epochs",
                [(f"epoch {record.epoch}", epoch_string(record)) for record in description.epochs],
            )
        )
    if description.show_runs and description.evaluations:
        multiple_evaluations = len(description.evaluations) > 1
        for evaluation in description.evaluations:
            if not evaluation.runtime.runs:
                continue
            runtime = evaluation.runtime
            sections.append(
                (
                    (
                        "runs"
                        if not multiple_evaluations
                        else f"runs {runtime.evaluation_id} {runtime.delay_seconds}s"
                    ),
                    [
                        (f"run {index}", evaluation_run_string(run))
                        for index, run in enumerate(evaluation.runtime.runs, start=1)
                    ],
                )
            )
    return sections


def feature_prerequisites_string(prerequisites: FeaturePrerequisites) -> str:
    return f"history={prerequisites.history_seconds}s warmup={prerequisites.warmup_rows} rows"


def epoch_string(record: TrainingEpochRecord) -> str:
    return (
        f"train={metric_bundle_string([], record.train_metrics.values)} "
        f"val={metric_bundle_string([], record.validation_metrics.values)}"
    )


def evaluation_run_string(run) -> str:
    metadata = " ".join(f"{key}={value}" for key, value in run.metadata.items())
    prefix = f"{metadata} " if metadata else ""
    return f"{prefix}events={run.n_events} metrics={metric_bundle_string([], run.metrics)}"

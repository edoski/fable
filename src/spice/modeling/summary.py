# pyright: strict

"""Modeling-owned workflow summary builders."""

from __future__ import annotations

from ..core.rendering import metric_fields, window_metric_fields
from .results import LoadedEvaluationSummary, LoadedTrainingSummary


def training_summary_sections(
    summary: LoadedTrainingSummary,
) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = summary.manifest
    runtime = summary.runtime
    return [
        (
            "dataset",
            [
                ("name", manifest.dataset_name),
                ("storage id", manifest.dataset_id),
                ("chain", manifest.chain_name),
                ("model", manifest.model.id),
                ("problem", manifest.problem_id),
                ("prediction", manifest.prediction_id),
            ],
        ),
        (
            "provenance",
            [
                ("artifact id", manifest.artifact_id),
                ("variant", manifest.variant.value),
                *([] if manifest.study is None else [("study", manifest.study.name)]),
                ("capability", f"{manifest.max_delay_seconds}s"),
            ],
        ),
        (
            "runtime",
            [
                ("lookback", f"{manifest.lookback_seconds}s"),
                ("best epoch", str(runtime.best_epoch)),
                ("device", runtime.resolved_device),
                ("precision", runtime.resolved_precision),
                ("compile", "on" if runtime.compiled else "off"),
                ("representation", manifest.representation_id),
                ("loader", runtime.loader_strategy_id),
                ("input storage", runtime.input_storage_mode_id),
                ("target storage", runtime.target_storage_mode_id),
                ("batch planner", runtime.batch_planner_id),
            ],
        ),
        (
            "metrics",
            [
                (
                    "split sizes",
                    (
                        f"train={runtime.split_sizes.train_samples:,} "
                        f"validation={runtime.split_sizes.validation_samples:,} "
                        f"test={runtime.split_sizes.test_samples:,}"
                    ),
                ),
                *[
                    (f"validation {label}", value)
                    for label, value in metric_fields(
                        manifest.training_metric_descriptors,
                        runtime.best_validation_metrics.values,
                    )
                ],
                *[
                    (f"test {label}", value)
                    for label, value in metric_fields(
                        manifest.training_metric_descriptors,
                        runtime.test_metrics.values,
                    )
                ],
            ],
        ),
    ]


def evaluation_summary_sections(
    summary: LoadedEvaluationSummary,
) -> list[tuple[str, list[tuple[str, str]]]]:
    manifest = summary.manifest
    runtime = summary.runtime
    return [
        (
            "dataset",
            [
                ("name", manifest.dataset_name),
                ("storage id", manifest.dataset_id),
                ("chain", manifest.chain_name),
                ("model", manifest.model.id),
                ("problem", manifest.problem_id),
                ("prediction", manifest.prediction_id),
            ],
        ),
        (
            "provenance",
            [
                ("artifact id", manifest.artifact_id),
                ("evaluation id", summary.evaluation_id),
                ("variant", manifest.variant.value),
                *([] if manifest.study is None else [("study", manifest.study.name)]),
                ("capability", f"{manifest.max_delay_seconds}s"),
                ("requested", f"{runtime.delay_seconds}s"),
            ],
        ),
        (
            "evaluation",
            [
                ("evaluator", runtime.evaluator_id),
                ("events", f"{runtime.total_events:,}"),
            ],
        ),
        (
            "results",
            metric_fields(runtime.metric_descriptors, runtime.metrics.values),
        ),
        *(
            []
            if not runtime.window_metrics
            else [
                (
                    "window metrics",
                    window_metric_fields(
                        runtime.metric_descriptors,
                        runtime.window_metrics,
                    ),
                )
            ]
        ),
    ]

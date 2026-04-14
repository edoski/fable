"""Modeling-owned workflow summary builders."""

from __future__ import annotations

from ..prediction import MetricDescriptor
from .results import SimulationSummaryRecord, TrainingSummary


def _metric_lines(
    descriptors: list[MetricDescriptor],
    metrics,
) -> list[tuple[str, str]]:
    return [
        (descriptor.label, f"{metrics.require(descriptor.id):.4f}")
        for descriptor in descriptors
        if descriptor.id in metrics.values
    ]


def _window_metric_lines(summary: SimulationSummaryRecord) -> list[tuple[str, str]]:
    return [
        (
            descriptor.label,
            (
                f"mean={summary.window_metrics[descriptor.id].mean:.4f} "
                f"std={summary.window_metrics[descriptor.id].std:.4f}"
            ),
        )
        for descriptor in summary.simulation_metric_descriptors
        if descriptor.id in summary.window_metrics
    ]


def training_summary_sections(
    summary: TrainingSummary,
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "dataset",
            [
                ("name", summary.dataset_name),
                ("storage id", summary.dataset_id),
                ("chain", summary.chain),
                ("model", summary.model_id),
                ("problem", summary.problem_id),
                ("prediction", summary.prediction_id),
            ],
        ),
        (
            "provenance",
            [
                ("artifact id", summary.artifact_id),
                ("variant", summary.variant.value),
                *([] if summary.study is None else [("study", summary.study.name)]),
                ("capability", f"{summary.max_supported_delay_seconds}s"),
            ],
        ),
        (
            "runtime",
            [
                ("lookback", f"{summary.lookback_seconds}s"),
                ("best epoch", str(summary.best_epoch)),
                ("device", summary.resolved_device),
                ("precision", summary.resolved_precision),
                ("compile", "on" if summary.compiled else "off"),
                ("representation", summary.representation_id),
                ("storage mode", summary.storage_mode_id),
                ("batch planner", summary.batch_planner_id),
            ],
        ),
        (
            "metrics",
            [
                (
                    "split sizes",
                    (
                        f"train={summary.split_sizes.train_samples:,} "
                        f"validation={summary.split_sizes.validation_samples:,} "
                        f"test={summary.split_sizes.test_samples:,}"
                    ),
                ),
                *[
                    (f"validation {label}", value)
                    for label, value in _metric_lines(
                        summary.training_metric_descriptors,
                        summary.best_validation_metrics,
                    )
                ],
                *[
                    (f"test {label}", value)
                    for label, value in _metric_lines(
                        summary.training_metric_descriptors,
                        summary.test_metrics,
                    )
                ],
            ],
        ),
    ]


def simulation_summary_sections(
    summary: SimulationSummaryRecord,
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "dataset",
            [
                ("name", summary.dataset_name),
                ("storage id", summary.dataset_id),
                ("chain", summary.chain),
                ("model", summary.model_id),
                ("problem", summary.problem_id),
                ("prediction", summary.prediction_id),
            ],
        ),
        (
            "provenance",
            [
                ("artifact id", summary.artifact_id),
                ("variant", summary.variant.value),
                *([] if summary.study is None else [("study", summary.study.name)]),
                ("capability", f"{summary.max_supported_delay_seconds}s"),
                ("requested", f"{summary.requested_delay_seconds}s"),
            ],
        ),
        (
            "simulation",
            [
                ("window", f"{summary.simulation_window_seconds}s"),
                ("repetitions", str(summary.repetitions)),
                ("events", f"{summary.total_events:,}"),
            ],
        ),
        (
            "results",
            _metric_lines(summary.simulation_metric_descriptors, summary.metrics),
        ),
        *(
            []
            if not summary.window_metrics
            else [("window metrics", _window_metric_lines(summary))]
        ),
    ]

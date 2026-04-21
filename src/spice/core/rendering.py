"""Small shared rendering helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..prediction import MetricDescriptor, WindowMetricSummary


def value_string(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def mapping_fields(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    return [(str(key).replace("_", " "), value_string(value)) for key, value in payload.items()]


def metric_string(value: float) -> str:
    return f"{value:.4f}"


def metric_bundle_string(
    descriptors: Sequence[MetricDescriptor],
    metrics: Mapping[str, float],
) -> str:
    if descriptors:
        ordered = [
            f"{descriptor.id}={metric_string(metrics[descriptor.id])}"
            for descriptor in descriptors
            if descriptor.id in metrics
        ]
        if ordered:
            return " ".join(ordered)
    return " ".join(
        f"{metric_id}={metric_string(value)}" for metric_id, value in sorted(metrics.items())
    )


def window_metric_fields(
    descriptors: Sequence[MetricDescriptor],
    metrics: Mapping[str, WindowMetricSummary],
) -> list[tuple[str, str]]:
    return [
        (
            descriptor.label,
            (
                f"mean={metric_string(metrics[descriptor.id].mean)} "
                f"std={metric_string(metrics[descriptor.id].std)}"
            ),
        )
        for descriptor in descriptors
        if descriptor.id in metrics
    ]

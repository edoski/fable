"""Evaluation-private Temporal Replay metric catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..metrics import MetricDescriptor, WindowMetricSummary


@dataclass(frozen=True, slots=True)
class TemporalReplayMetricSpec:
    id: str
    label: str
    role: Literal["primary", "secondary", "diagnostic"]
    aggregate: Literal["event_mean", "fee_sum"]
    direction: Literal["maximize", "minimize"] | None = None
    window_summary: bool = False

    def descriptor(self) -> MetricDescriptor:
        return MetricDescriptor(
            id=self.id,
            label=self.label,
            role=self.role,
            direction=self.direction,
        )


TEMPORAL_REPLAY_METRICS: tuple[TemporalReplayMetricSpec, ...] = (
    TemporalReplayMetricSpec(
        id="profit_over_baseline",
        label="profit over baseline",
        role="primary",
        direction="maximize",
        aggregate="event_mean",
        window_summary=True,
    ),
    TemporalReplayMetricSpec(
        id="cost_over_optimum",
        label="cost over optimum",
        role="secondary",
        direction="minimize",
        aggregate="event_mean",
        window_summary=True,
    ),
    TemporalReplayMetricSpec(
        id="baseline_cost_over_optimum",
        label="baseline cost over optimum",
        role="secondary",
        direction="minimize",
        aggregate="event_mean",
        window_summary=True,
    ),
    TemporalReplayMetricSpec(
        id="exact_optimum_hit_rate",
        label="exact optimum hit rate",
        role="secondary",
        direction="maximize",
        aggregate="event_mean",
        window_summary=True,
    ),
    TemporalReplayMetricSpec(
        id="realized_fee_sum",
        label="realized fee sum",
        role="diagnostic",
        aggregate="fee_sum",
    ),
    TemporalReplayMetricSpec(
        id="baseline_fee_sum",
        label="baseline fee sum",
        role="diagnostic",
        aggregate="fee_sum",
    ),
    TemporalReplayMetricSpec(
        id="optimum_fee_sum",
        label="optimum fee sum",
        role="diagnostic",
        aggregate="fee_sum",
    ),
)

TEMPORAL_REPLAY_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = tuple(
    metric.descriptor() for metric in TEMPORAL_REPLAY_METRICS
)
_TEMPORAL_REPLAY_EVENT_MEAN_METRICS: tuple[TemporalReplayMetricSpec, ...] = tuple(
    metric for metric in TEMPORAL_REPLAY_METRICS if metric.aggregate == "event_mean"
)
_TEMPORAL_REPLAY_FEE_SUM_METRICS: tuple[TemporalReplayMetricSpec, ...] = tuple(
    metric for metric in TEMPORAL_REPLAY_METRICS if metric.aggregate == "fee_sum"
)
_TEMPORAL_REPLAY_WINDOW_METRICS: tuple[TemporalReplayMetricSpec, ...] = tuple(
    metric for metric in TEMPORAL_REPLAY_METRICS if metric.window_summary
)
_TEMPORAL_REPLAY_METRIC_IDS = frozenset(metric.id for metric in TEMPORAL_REPLAY_METRICS)
_TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS = frozenset(
    metric.id for metric in _TEMPORAL_REPLAY_EVENT_MEAN_METRICS
)
_TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS = frozenset(
    metric.id for metric in _TEMPORAL_REPLAY_FEE_SUM_METRICS
)
_TEMPORAL_REPLAY_WINDOW_METRIC_IDS = frozenset(
    metric.id for metric in _TEMPORAL_REPLAY_WINDOW_METRICS
)


def temporal_replay_fee_sums(
    *,
    realized_fee_sum: float,
    baseline_fee_sum: float,
    optimum_fee_sum: float,
) -> dict[str, float]:
    return _require_ids(
        {
            "realized_fee_sum": realized_fee_sum,
            "baseline_fee_sum": baseline_fee_sum,
            "optimum_fee_sum": optimum_fee_sum,
        },
        expected_ids=_TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS,
        label="Temporal Replay fee-sum metrics",
    )


def temporal_replay_event_mean_values(
    event_metric_sums: Mapping[str, float],
    *,
    n_events: int,
) -> dict[str, float]:
    if n_events <= 0:
        raise ValueError("evaluation event count must be positive")
    sums = _require_ids(
        event_metric_sums,
        expected_ids=_TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS,
        label="Temporal Replay event metric sums",
    )
    return {metric_id: metric_sum / n_events for metric_id, metric_sum in sums.items()}


def temporal_replay_event_sum_totals(
    run_event_sums: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    totals = {metric_id: 0.0 for metric_id in _TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS}
    for event_sums in run_event_sums:
        values = _require_ids(
            event_sums,
            expected_ids=_TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS,
            label="Temporal Replay event metric sums",
        )
        for metric_id in totals:
            totals[metric_id] += values[metric_id]
    return _require_ids(
        totals,
        expected_ids=_TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS,
        label="Temporal Replay event metric sums",
    )


def temporal_replay_metric_values(
    *,
    event_metric_sums: Mapping[str, float],
    fee_sums: Mapping[str, float],
    n_events: int,
) -> dict[str, float]:
    values = {
        **temporal_replay_event_mean_values(event_metric_sums, n_events=n_events),
        **_require_ids(
            fee_sums,
            expected_ids=_TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS,
            label="Temporal Replay fee-sum metrics",
        ),
    }
    return validate_temporal_replay_metric_values(values)


def temporal_replay_fee_sum_totals(
    run_fee_sums: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    totals = {metric_id: 0.0 for metric_id in _TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS}
    for fee_sums in run_fee_sums:
        values = _require_ids(
            fee_sums,
            expected_ids=_TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS,
            label="Temporal Replay fee-sum metrics",
        )
        for metric_id in totals:
            totals[metric_id] += values[metric_id]
    return _require_ids(
        totals,
        expected_ids=_TEMPORAL_REPLAY_FEE_SUM_METRIC_IDS,
        label="Temporal Replay fee-sum metrics",
    )


def validate_temporal_replay_metric_values(values: Mapping[str, float]) -> dict[str, float]:
    return _require_ids(
        values,
        expected_ids=_TEMPORAL_REPLAY_METRIC_IDS,
        label="Temporal Replay metric values",
    )


def temporal_replay_window_metrics(
    run_metrics: Iterable[Mapping[str, float]],
) -> dict[str, WindowMetricSummary]:
    collected: dict[str, list[float]] = {
        metric_id: [] for metric_id in sorted(_TEMPORAL_REPLAY_WINDOW_METRIC_IDS)
    }
    for metrics in run_metrics:
        values = validate_temporal_replay_metric_values(metrics)
        for metric_id in collected:
            collected[metric_id].append(values[metric_id])
    return {
        metric_id: WindowMetricSummary(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
        )
        for metric_id, values in collected.items()
    }


def _require_ids(
    values: Mapping[str, float],
    *,
    expected_ids: frozenset[str],
    label: str,
) -> dict[str, float]:
    value_ids = set(values)
    missing = expected_ids - value_ids
    extra = value_ids - expected_ids
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"extra: {', '.join(sorted(extra))}")
        raise ValueError(f"{label} ids do not match catalog ({'; '.join(parts)})")
    return {metric_id: float(values[metric_id]) for metric_id in sorted(expected_ids)}

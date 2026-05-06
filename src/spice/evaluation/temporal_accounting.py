"""Temporal decision accounting used by the Temporal Replay Runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..prediction.decoded_offsets import DecodedOffsets
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from ._temporal_replay_metric_catalog import (
    TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS,
    TEMPORAL_REPLAY_EVENT_MEAN_METRICS,
    TEMPORAL_REPLAY_WINDOW_METRICS,
)
from .temporal_replay_results import (
    TemporalReplayEventMetricSums,
    TemporalReplayMetrics,
    TemporalReplayResult,
    TemporalReplayRunResult,
    TemporalReplayWindowMetric,
)

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _TemporalCostSummary:
    n_events: int
    realized_fee_sum: float
    baseline_fee_sum: float
    optimum_fee_sum: float
    event_metric_sums: TemporalReplayEventMetricSums


def summarize_selected_temporal_decisions(
    store: CompiledProblemStore,
    execution_policy: CompiledExecutionPolicyContract,
    decoded_offsets: DecodedOffsets,
    action_space: PreparedActionSpace,
    selected_positions: IntVector,
    *,
    metadata: dict[str, str | int | float],
) -> TemporalReplayRunResult:
    realized = execution_policy.realize_selections(
        store,
        decoded_offsets,
        action_space,
        selected_positions,
    )
    realized_logs = store.log_base_fees[realized.realized_rows]
    realized_fees = np.exp(realized_logs.astype(np.float64, copy=False))
    baseline_fees = np.exp(
        store.log_base_fees[realized.baseline_rows].astype(np.float64, copy=False)
    )
    realized_total = float(realized_fees.sum())
    baseline_total = float(baseline_fees.sum())
    optimum_logs = store.log_base_fees[realized.optimum_rows].astype(np.float64, copy=False)
    optimum_fees = np.exp(optimum_logs)
    optimum_total = float(optimum_fees.sum())
    if baseline_total <= 0.0:
        raise ValueError("baseline fee total must be positive")
    if optimum_total <= 0.0:
        raise ValueError("optimum fee total must be positive")
    if np.any(baseline_fees <= 0.0):
        raise ValueError("baseline fees must be positive")
    if np.any(optimum_fees <= 0.0):
        raise ValueError("optimum fees must be positive")

    profit_values = (baseline_fees - realized_fees) / baseline_fees
    cost_values = (realized_fees - optimum_fees) / optimum_fees
    baseline_cost_values = (baseline_fees - optimum_fees) / optimum_fees
    exact_hits = realized.realized_rows == realized.optimum_rows
    costs = _TemporalCostSummary(
        n_events=int(selected_positions.shape[0]),
        realized_fee_sum=realized_total,
        baseline_fee_sum=baseline_total,
        optimum_fee_sum=optimum_total,
        event_metric_sums=TemporalReplayEventMetricSums(
            profit_over_baseline=float(profit_values.sum()),
            cost_over_optimum=float(cost_values.sum()),
            baseline_cost_over_optimum=float(baseline_cost_values.sum()),
            exact_optimum_hit_rate=float(exact_hits.sum()),
        ),
    )
    return TemporalReplayRunResult(
        n_events=costs.n_events,
        metrics=_event_mean_metrics(costs),
        event_metric_sums=costs.event_metric_sums,
        metadata={
            **dict(metadata),
            "overflow_count": int(realized.overflow_mask.sum()),
        },
    )


def summarize_temporal_accounting_runs(
    runs: list[TemporalReplayRunResult],
) -> TemporalReplayResult:
    if not runs:
        raise ValueError("evaluation produced no runs")

    realized_fee_sum = sum(run.metrics.realized_fee_sum for run in runs)
    baseline_fee_sum = sum(run.metrics.baseline_fee_sum for run in runs)
    optimum_fee_sum = sum(run.metrics.optimum_fee_sum for run in runs)
    if baseline_fee_sum <= 0.0:
        raise ValueError("baseline fee sum must be positive")
    if optimum_fee_sum <= 0.0:
        raise ValueError("optimum fee sum must be positive")
    total_events = sum(run.n_events for run in runs)
    costs = _TemporalCostSummary(
        n_events=total_events,
        realized_fee_sum=realized_fee_sum,
        baseline_fee_sum=baseline_fee_sum,
        optimum_fee_sum=optimum_fee_sum,
        event_metric_sums=TemporalReplayEventMetricSums.from_values(
            {
                metric.field_name: sum(
                    run.event_metric_sums.value(metric.id) for run in runs
                )
                for metric in TEMPORAL_REPLAY_EVENT_MEAN_METRICS
            }
        ),
    )
    return TemporalReplayResult(
        metrics=_event_mean_metrics(costs),
        window_metrics=(
            {
                metric.id: _summarize_window_metric(
                    [getattr(run.metrics, metric.field_name) for run in runs]
                )
                for metric in TEMPORAL_REPLAY_WINDOW_METRICS
            }
            if len(runs) > 1
            else {}
        ),
        total_events=total_events,
        runs=tuple(runs),
    )


def _summarize_window_metric(values: list[float]) -> TemporalReplayWindowMetric:
    return TemporalReplayWindowMetric(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def _event_mean_metrics(costs: _TemporalCostSummary) -> TemporalReplayMetrics:
    ratio_means = {
        metric.field_name: _event_metric_mean(
            costs.event_metric_sums,
            metric.id,
            costs.n_events,
        )
        for metric in TEMPORAL_REPLAY_EVENT_MEAN_METRICS
    }
    return TemporalReplayMetrics(
        profit_over_baseline=ratio_means["profit_over_baseline"],
        cost_over_optimum=ratio_means["cost_over_optimum"],
        baseline_cost_over_optimum=ratio_means["baseline_cost_over_optimum"],
        exact_optimum_hit_rate=ratio_means["exact_optimum_hit_rate"],
        realized_fee_sum=costs.realized_fee_sum,
        baseline_fee_sum=costs.baseline_fee_sum,
        optimum_fee_sum=costs.optimum_fee_sum,
    )


def _event_metric_mean(
    event_metric_sums: TemporalReplayEventMetricSums,
    metric_id: str,
    n_events: int,
) -> float:
    if n_events <= 0:
        raise ValueError("evaluation event count must be positive")
    if metric_id not in TEMPORAL_REPLAY_EVENT_MEAN_METRIC_IDS:
        raise KeyError(metric_id)
    return event_metric_sums.value(metric_id) / n_events

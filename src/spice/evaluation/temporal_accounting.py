"""Temporal decision accounting shared by evaluator adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..prediction.decoded_offsets import DecodedOffsets
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore
from .contracts import IntVector
from .temporal_replay_results import (
    BASELINE_COST_OVER_OPTIMUM,
    COST_OVER_OPTIMUM,
    EXACT_OPTIMUM_HIT_RATE,
    PROFIT_OVER_BASELINE,
    TEMPORAL_RATIO_METRIC_IDS,
    TemporalReplayEventMetricSums,
    TemporalReplayMetrics,
    TemporalReplayResult,
    TemporalReplayRunResult,
    TemporalReplayWindowMetric,
)

_WINDOW_METRIC_IDS = (
    PROFIT_OVER_BASELINE,
    COST_OVER_OPTIMUM,
    BASELINE_COST_OVER_OPTIMUM,
    EXACT_OPTIMUM_HIT_RATE,
)


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
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    metadata: dict[str, str | int | float],
) -> TemporalReplayRunResult:
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    realized = execution_policy.realize_selections(
        store,
        decoded_offsets,
        sample_indices,
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
        event_metric_sums=TemporalReplayEventMetricSums(
            profit_over_baseline=sum(
                run.event_metric_sums.profit_over_baseline for run in runs
            ),
            cost_over_optimum=sum(run.event_metric_sums.cost_over_optimum for run in runs),
            baseline_cost_over_optimum=sum(
                run.event_metric_sums.baseline_cost_over_optimum for run in runs
            ),
            exact_optimum_hit_rate=sum(
                run.event_metric_sums.exact_optimum_hit_rate for run in runs
            ),
        ),
    )
    return TemporalReplayResult(
        metrics=_event_mean_metrics(costs),
        window_metrics={
            "profit_over_baseline": _summarize_window_metric(
                [run.metrics.profit_over_baseline for run in runs]
            ),
            "cost_over_optimum": _summarize_window_metric(
                [run.metrics.cost_over_optimum for run in runs]
            ),
            "baseline_cost_over_optimum": _summarize_window_metric(
                [run.metrics.baseline_cost_over_optimum for run in runs]
            ),
            "exact_optimum_hit_rate": _summarize_window_metric(
                [run.metrics.exact_optimum_hit_rate for run in runs]
            ),
        }
        if len(runs) > 1
        else {},
        total_events=total_events,
        runs=tuple(runs),
    )


def _summarize_window_metric(values: list[float]) -> TemporalReplayWindowMetric:
    return TemporalReplayWindowMetric(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def _event_mean_metrics(costs: _TemporalCostSummary) -> TemporalReplayMetrics:
    return TemporalReplayMetrics(
        profit_over_baseline=_event_metric_mean(
            costs.event_metric_sums,
            PROFIT_OVER_BASELINE,
            costs.n_events,
        ),
        cost_over_optimum=_event_metric_mean(
            costs.event_metric_sums,
            COST_OVER_OPTIMUM,
            costs.n_events,
        ),
        baseline_cost_over_optimum=_event_metric_mean(
            costs.event_metric_sums,
            BASELINE_COST_OVER_OPTIMUM,
            costs.n_events,
        ),
        exact_optimum_hit_rate=_event_metric_mean(
            costs.event_metric_sums,
            EXACT_OPTIMUM_HIT_RATE,
            costs.n_events,
        ),
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
    if metric_id not in TEMPORAL_RATIO_METRIC_IDS:
        raise KeyError(metric_id)
    return event_metric_sums.value(metric_id) / n_events

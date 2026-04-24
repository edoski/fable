"""Shared evaluator sampling and cost summarization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..prediction.base import MetricSet, WindowMetricSummary
from ..prediction.contracts import DecodedPredictionResult, require_decoded_offsets
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from .config import EvaluationAggregation
from .contracts import EvaluationMetadataValue, EvaluationRun, EvaluationSummary, IntVector


@dataclass(frozen=True, slots=True)
class ChronologicalSampleView:
    sample_positions: IntVector
    sample_timestamps: IntVector


@dataclass(frozen=True, slots=True)
class CandidateWindowSummary:
    anchor_rows: IntVector
    baseline_rows: IntVector
    candidate_end_rows: IntVector
    candidate_counts: IntVector
    last_candidate_rows: IntVector
    optimum_rows: IntVector


def sample_poisson_arrivals(
    rng: np.random.Generator,
    *,
    rate_per_second: float,
    start_timestamp: float,
    end_timestamp: float,
) -> NDArray[np.float64]:
    if rate_per_second <= 0:
        raise ValueError("rate_per_second must be positive")
    arrivals: list[float] = []
    cursor = start_timestamp
    while cursor < end_timestamp:
        cursor += rng.exponential(1.0 / rate_per_second)
        if cursor < end_timestamp:
            arrivals.append(cursor)
    return np.asarray(arrivals, dtype=np.float64)


def chronological_sample_view(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> ChronologicalSampleView:
    resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
    sample_timestamps = store.timestamps[store.anchor_rows[resolved_sample_indices]].astype(
        np.int64,
        copy=False,
    )
    order = np.argsort(sample_timestamps, kind="stable").astype(np.int64, copy=False)
    return ChronologicalSampleView(
        sample_positions=order,
        sample_timestamps=sample_timestamps[order],
    )


def select_sample_positions_for_arrivals(
    sample_timestamps: NDArray[np.int64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(sample_timestamps, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


def candidate_window_summary(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> CandidateWindowSummary:
    resolved_indices = sample_indices.astype(np.int64, copy=False)
    anchor_rows = store.anchor_rows[resolved_indices].astype(np.int64, copy=False)
    baseline_rows = store.candidate_start_rows[resolved_indices].astype(np.int64, copy=False)
    candidate_end_rows = store.candidate_end_rows[resolved_indices].astype(np.int64, copy=False)
    candidate_counts = (candidate_end_rows - baseline_rows).astype(np.int64, copy=False)
    if np.any(candidate_counts <= 0):
        raise ValueError("evaluation requires at least one candidate row per sample")
    last_candidate_rows = (candidate_end_rows - 1).astype(np.int64, copy=False)
    optimum_rows = np.empty(resolved_indices.shape[0], dtype=np.int64)
    for row, (start_row, end_row) in enumerate(
        zip(baseline_rows, candidate_end_rows, strict=True)
    ):
        optimum_rows[row] = int(start_row + np.argmin(store.log_base_fees[start_row:end_row]))
    return CandidateWindowSummary(
        anchor_rows=anchor_rows,
        baseline_rows=baseline_rows,
        candidate_end_rows=candidate_end_rows,
        candidate_counts=candidate_counts,
        last_candidate_rows=last_candidate_rows,
        optimum_rows=optimum_rows,
    )


def single_run_summary(
    *,
    metric_values: dict[str, float],
    n_events: int,
    metadata: dict[str, EvaluationMetadataValue],
) -> EvaluationSummary:
    run = EvaluationRun(
        n_events=n_events,
        metrics=metric_values,
        metadata=metadata,
    )
    return EvaluationSummary(
        metrics=MetricSet(values=dict(metric_values)),
        window_metrics={},
        total_events=n_events,
        runs=[run],
    )


def summarize_selected_costs(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_result: DecodedPredictionResult,
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    aggregation: EvaluationAggregation = EvaluationAggregation.TOTAL_RATIO,
    metadata: dict[str, str | int | float],
) -> EvaluationRun:
    decoded_offsets = require_decoded_offsets(decoded_result)
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    realized = realization_policy.realize_selections(
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
    if aggregation is EvaluationAggregation.EVENT_MEAN:
        profit_over_baseline = float(profit_values.mean())
        cost_over_optimum = float(cost_values.mean())
        baseline_cost_over_optimum = float(baseline_cost_values.mean())
    else:
        profit_over_baseline = (baseline_total - realized_total) / baseline_total
        cost_over_optimum = (realized_total - optimum_total) / optimum_total
        baseline_cost_over_optimum = (baseline_total - optimum_total) / optimum_total

    return EvaluationRun(
        n_events=int(selected_positions.shape[0]),
        metrics={
            "profit_over_baseline": profit_over_baseline,
            "cost_over_optimum": cost_over_optimum,
            "baseline_cost_over_optimum": baseline_cost_over_optimum,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
        },
        metadata={
            **dict(metadata),
            "overflow_count": int(realized.overflow_mask.sum()),
        },
        event_metric_sums={
            "profit_over_baseline": float(profit_values.sum()),
            "cost_over_optimum": float(cost_values.sum()),
            "baseline_cost_over_optimum": float(baseline_cost_values.sum()),
        },
    )


def summarize_runs(
    runs: list[EvaluationRun],
    *,
    aggregation: EvaluationAggregation = EvaluationAggregation.TOTAL_RATIO,
) -> EvaluationSummary:
    if not runs:
        raise ValueError("evaluation produced no runs")

    realized_fee_sum = sum(run.metrics["realized_fee_sum"] for run in runs)
    baseline_fee_sum = sum(run.metrics["baseline_fee_sum"] for run in runs)
    optimum_fee_sum = sum(run.metrics["optimum_fee_sum"] for run in runs)
    if baseline_fee_sum <= 0.0:
        raise ValueError("baseline fee sum must be positive")
    if optimum_fee_sum <= 0.0:
        raise ValueError("optimum fee sum must be positive")
    total_events = sum(run.n_events for run in runs)
    if aggregation is EvaluationAggregation.EVENT_MEAN:
        profit_over_baseline = _event_metric_mean(runs, "profit_over_baseline", total_events)
        cost_over_optimum = _event_metric_mean(runs, "cost_over_optimum", total_events)
        baseline_cost_over_optimum = _event_metric_mean(
            runs,
            "baseline_cost_over_optimum",
            total_events,
        )
    else:
        profit_over_baseline = (baseline_fee_sum - realized_fee_sum) / baseline_fee_sum
        cost_over_optimum = (realized_fee_sum - optimum_fee_sum) / optimum_fee_sum
        baseline_cost_over_optimum = (baseline_fee_sum - optimum_fee_sum) / optimum_fee_sum
    return EvaluationSummary(
        metrics=MetricSet(
            values={
                "profit_over_baseline": profit_over_baseline,
                "cost_over_optimum": cost_over_optimum,
                "baseline_cost_over_optimum": baseline_cost_over_optimum,
                "realized_fee_sum": realized_fee_sum,
                "baseline_fee_sum": baseline_fee_sum,
                "optimum_fee_sum": optimum_fee_sum,
            }
        ),
        window_metrics={
            "profit_over_baseline": _summarize_window_metric(
                [run.metrics["profit_over_baseline"] for run in runs]
            ),
            "cost_over_optimum": _summarize_window_metric(
                [run.metrics["cost_over_optimum"] for run in runs]
            ),
            "baseline_cost_over_optimum": _summarize_window_metric(
                [run.metrics["baseline_cost_over_optimum"] for run in runs]
            ),
        }
        if len(runs) > 1
        else {},
        total_events=total_events,
        runs=runs,
    )


def _event_metric_mean(
    runs: list[EvaluationRun],
    metric_id: str,
    total_events: int,
) -> float:
    if total_events <= 0:
        raise ValueError("evaluation event count must be positive")
    return sum(run.event_metric_sums[metric_id] for run in runs) / total_events


def _summarize_window_metric(values: list[float]) -> WindowMetricSummary:
    return WindowMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )

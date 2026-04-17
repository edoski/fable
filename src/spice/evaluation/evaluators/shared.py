"""Shared evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ...prediction.base import MetricDescriptor, MetricSet, WindowMetricSummary
from ...temporal.problem_store import CompiledProblemStore
from ..base import EvaluationRun, EvaluationSummary

IntVector = NDArray[np.int64]

EVALUATION_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        id="profit_over_baseline",
        label="profit over baseline",
        role="primary",
    ),
    MetricDescriptor(
        id="cost_over_optimum",
        label="cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="baseline_cost_over_optimum",
        label="baseline cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="realized_fee_sum",
        label="realized fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="baseline_fee_sum",
        label="baseline fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="optimum_fee_sum",
        label="optimum fee sum",
        role="diagnostic",
    ),
)


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


def select_sample_positions_for_arrivals(
    sample_timestamps: NDArray[np.int64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(sample_timestamps, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


def summarize_selected_costs(
    store: CompiledProblemStore,
    decoded_offsets: Sequence[int],
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    metadata: dict[str, str | int | float],
) -> EvaluationRun:
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    selected_sample_indices = sample_indices[selected_positions]
    selected_offsets = np.asarray(decoded_offsets, dtype=np.int64)[selected_positions]
    selected_anchor_rows = store.anchor_rows[selected_sample_indices]
    candidate_starts = selected_anchor_rows + 1
    candidate_ends = store.candidate_end_rows[selected_sample_indices]
    realized_rows = candidate_starts + selected_offsets
    realized_logs = store.log_base_fees[realized_rows]
    realized_total = float(np.exp(realized_logs.astype(np.float64, copy=False)).sum())
    baseline_total = float(
        np.exp(store.log_base_fees[candidate_starts].astype(np.float64, copy=False)).sum()
    )
    optimum_logs = np.empty(selected_sample_indices.shape[0], dtype=np.float64)
    for index, (start_row, end_row) in enumerate(
        zip(candidate_starts, candidate_ends, strict=True)
    ):
        optimum_logs[index] = float(store.log_base_fees[start_row:end_row].min())
    optimum_total = float(np.exp(optimum_logs).sum())

    return EvaluationRun(
        n_events=int(selected_positions.shape[0]),
        metrics={
            "profit_over_baseline": (baseline_total - realized_total) / baseline_total,
            "cost_over_optimum": (realized_total - optimum_total) / optimum_total,
            "baseline_cost_over_optimum": (baseline_total - optimum_total) / optimum_total,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
        },
        metadata=dict(metadata),
    )


def summarize_runs(runs: list[EvaluationRun]) -> EvaluationSummary:
    if not runs:
        raise ValueError("evaluation produced no runs")

    realized_fee_sum = sum(run.metrics["realized_fee_sum"] for run in runs)
    baseline_fee_sum = sum(run.metrics["baseline_fee_sum"] for run in runs)
    optimum_fee_sum = sum(run.metrics["optimum_fee_sum"] for run in runs)
    return EvaluationSummary(
        metrics=MetricSet(
            values={
                "profit_over_baseline": (baseline_fee_sum - realized_fee_sum)
                / baseline_fee_sum,
                "cost_over_optimum": (realized_fee_sum - optimum_fee_sum) / optimum_fee_sum,
                "baseline_cost_over_optimum": (baseline_fee_sum - optimum_fee_sum)
                / optimum_fee_sum,
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
        total_events=sum(run.n_events for run in runs),
        runs=runs,
    )


def _summarize_window_metric(values: list[float]) -> WindowMetricSummary:
    return WindowMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )

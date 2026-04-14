"""Shared replay helpers for offset-based prediction families."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.reporting import NullReporter, Reporter
from ...temporal.problem_store import CompiledProblemStore
from ..base import (
    MetricDescriptor,
    MetricSet,
    PredictionSimulationRun,
    PredictionSimulationSummary,
    WindowMetricSummary,
)

IntVector = NDArray[np.int64]

SIMULATION_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
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


def summarize_realized_costs(
    store: CompiledProblemStore,
    predicted_offsets: Sequence[int],
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    window_start_timestamp: float,
    window_end_timestamp: float,
    n_arrivals: int,
) -> PredictionSimulationRun:
    if len(predicted_offsets) != int(sample_indices.shape[0]):
        raise ValueError("predicted_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    selected_sample_indices = sample_indices[selected_positions]
    selected_offsets = np.asarray(predicted_offsets, dtype=np.int64)[selected_positions]
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

    return PredictionSimulationRun(
        window_start_timestamp=window_start_timestamp,
        window_end_timestamp=window_end_timestamp,
        n_arrivals=n_arrivals,
        n_events=int(selected_positions.shape[0]),
        metrics={
            "profit_over_baseline": (baseline_total - realized_total) / baseline_total,
            "cost_over_optimum": (realized_total - optimum_total) / optimum_total,
            "baseline_cost_over_optimum": (baseline_total - optimum_total) / optimum_total,
            "realized_fee_sum": realized_total,
            "baseline_fee_sum": baseline_total,
            "optimum_fee_sum": optimum_total,
        },
    )


def _summarize_window_metric(values: list[float]) -> WindowMetricSummary:
    return WindowMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def run_offset_replay(
    store: CompiledProblemStore,
    predicted_offsets: object,
    sample_indices: IntVector,
    *,
    family_id: str,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
    seed: int,
    reporter: Reporter | None = None,
) -> PredictionSimulationSummary:
    reporter = reporter or NullReporter()
    if not isinstance(predicted_offsets, list):
        raise TypeError(f"{family_id} prediction buffer must be a list")
    if len(predicted_offsets) != int(sample_indices.shape[0]):
        raise ValueError("predicted_offsets must align with sample_indices")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    sample_timestamps = store.timestamps[store.anchor_rows[sample_indices]]
    first_timestamp = int(sample_timestamps[0])
    last_timestamp = int(sample_timestamps[-1])
    latest_start = last_timestamp - window_seconds
    if latest_start < first_timestamp:
        raise ValueError("Evaluation examples do not cover the requested simulation window")

    rng = np.random.default_rng(seed)
    runs: list[PredictionSimulationRun] = []
    task_id = reporter.start_task(
        "simulate repetitions",
        total=repetitions,
        unit="repetitions",
    )
    for repetition in range(repetitions):
        window_start = float(rng.uniform(first_timestamp, latest_start))
        window_end = window_start + window_seconds
        arrivals = sample_poisson_arrivals(
            rng,
            rate_per_second=arrival_rate_per_second,
            start_timestamp=window_start,
            end_timestamp=window_end,
        )
        selected_positions = select_sample_positions_for_arrivals(sample_timestamps, arrivals)
        if selected_positions.size == 0:
            reporter.update_task(
                task_id,
                completed=repetition + 1,
                message="no valid arrivals",
            )
            continue
        summary = summarize_realized_costs(
            store,
            predicted_offsets,
            sample_indices,
            selected_positions,
            window_start_timestamp=window_start,
            window_end_timestamp=window_end,
            n_arrivals=int(arrivals.shape[0]),
        )
        runs.append(summary)
        reporter.update_task(
            task_id,
            completed=repetition + 1,
            message=f"events={summary.n_events}",
        )

    if not runs:
        raise ValueError("Simulation produced no valid runs")

    realized_fee_sum = sum(run.metrics["realized_fee_sum"] for run in runs)
    baseline_fee_sum = sum(run.metrics["baseline_fee_sum"] for run in runs)
    optimum_fee_sum = sum(run.metrics["optimum_fee_sum"] for run in runs)
    summary = PredictionSimulationSummary(
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
        },
        total_events=sum(run.n_events for run in runs),
        runs=runs,
    )
    reporter.finish_task(task_id, message=f"total_events={summary.total_events}")
    return summary

"""Economic simulation helpers for temporal evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.console import NullReporter, Reporter
from ..data.datasets import TemporalDatasetStore

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class SimulationRunSummary:
    window_start_timestamp: float
    window_end_timestamp: float
    n_arrivals: int
    n_events: int
    profit_over_baseline: float
    cost_over_optimum: float
    baseline_cost_over_optimum: float


@dataclass(slots=True)
class SimulationSummary:
    mean_profit_over_baseline: float
    std_profit_over_baseline: float
    mean_cost_over_optimum: float
    std_cost_over_optimum: float
    mean_baseline_cost_over_optimum: float
    std_baseline_cost_over_optimum: float
    total_events: int
    runs: list[SimulationRunSummary]


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
    anchor_timestamps: NDArray[np.int64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(anchor_timestamps, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


def summarize_realized_costs(
    store: TemporalDatasetStore,
    predicted_offsets: list[int],
    sample_indices: IntVector,
    selected_positions: IntVector,
    *,
    window_start_timestamp: float,
    window_end_timestamp: float,
    n_arrivals: int,
) -> SimulationRunSummary:
    if len(predicted_offsets) != int(sample_indices.shape[0]):
        raise ValueError("predicted_offsets must align with sample_indices")
    if selected_positions.size == 0:
        raise ValueError("selected_positions must be non-empty")

    selected_sample_indices = sample_indices[selected_positions]
    selected_offsets = np.asarray(predicted_offsets, dtype=np.int64)[selected_positions]
    selected_action_log_fees = store.action_log_fees[selected_sample_indices]
    realized_logs = selected_action_log_fees[np.arange(selected_offsets.shape[0]), selected_offsets]
    realized_total = float(np.exp(realized_logs.astype(np.float64, copy=False)).sum())
    baseline_total = float(
        np.exp(
            store.next_block_log_fee[selected_sample_indices].astype(np.float64, copy=False)
        ).sum()
    )
    optimum_total = float(
        np.exp(store.optimal_log_fee[selected_sample_indices].astype(np.float64, copy=False)).sum()
    )

    return SimulationRunSummary(
        window_start_timestamp=window_start_timestamp,
        window_end_timestamp=window_end_timestamp,
        n_arrivals=n_arrivals,
        n_events=int(selected_positions.shape[0]),
        profit_over_baseline=(baseline_total - realized_total) / baseline_total,
        cost_over_optimum=(realized_total - optimum_total) / optimum_total,
        baseline_cost_over_optimum=(baseline_total - optimum_total) / optimum_total,
    )


def run_temporal_simulation(
    store: TemporalDatasetStore,
    predicted_offsets: list[int],
    *,
    sample_indices: IntVector,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
    seed: int,
    reporter: Reporter | None = None,
) -> SimulationSummary:
    reporter = reporter or NullReporter()
    if len(predicted_offsets) != int(sample_indices.shape[0]):
        raise ValueError("predicted_offsets must align with sample_indices")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    anchor_timestamps = store.timestamps[store.anchor_row_indices[sample_indices]]
    first_timestamp = int(anchor_timestamps[0])
    last_timestamp = int(anchor_timestamps[-1])
    latest_start = last_timestamp - window_seconds
    if latest_start < first_timestamp:
        raise ValueError("Evaluation examples do not cover the requested simulation window")

    rng = np.random.default_rng(seed)
    runs: list[SimulationRunSummary] = []
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
        selected_positions = select_sample_positions_for_arrivals(anchor_timestamps, arrivals)
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

    profits = [run.profit_over_baseline for run in runs]
    model_costs = [run.cost_over_optimum for run in runs]
    baseline_costs = [run.baseline_cost_over_optimum for run in runs]
    summary = SimulationSummary(
        mean_profit_over_baseline=statistics.fmean(profits),
        std_profit_over_baseline=statistics.pstdev(profits),
        mean_cost_over_optimum=statistics.fmean(model_costs),
        std_cost_over_optimum=statistics.pstdev(model_costs),
        mean_baseline_cost_over_optimum=statistics.fmean(baseline_costs),
        std_baseline_cost_over_optimum=statistics.pstdev(baseline_costs),
        total_events=sum(run.n_events for run in runs),
        runs=runs,
    )
    reporter.finish_task(task_id, message=f"total_events={summary.total_events}")
    return summary

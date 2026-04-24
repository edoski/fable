"""Replay evaluator engine and sampler policies."""

from __future__ import annotations

import numpy as np

from ..core.errors import SpiceOperatorError
from ..prediction.contracts import DecodedPredictionResult, require_decoded_offsets
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from .config import EvaluationAggregation, EvaluatorConfig
from .contracts import EvaluationSummary, IntVector
from .shared import (
    chronological_sample_view,
    sample_poisson_arrivals,
    select_sample_positions_for_arrivals,
    summarize_runs,
    summarize_selected_costs,
)


def run_fullset(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_result: DecodedPredictionResult,
    sample_indices: IntVector,
    *,
    aggregation: EvaluationAggregation = EvaluationAggregation.TOTAL_RATIO,
) -> EvaluationSummary:
    decoded_offsets = require_decoded_offsets(decoded_result)
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    run = summarize_selected_costs(
        store,
        realization_policy,
        decoded_offsets,
        sample_indices,
        np.arange(sample_indices.shape[0], dtype=np.int64),
        aggregation=aggregation,
        metadata={"mode": "fullset"},
    )
    return summarize_runs([run], aggregation=aggregation)


def run_uniform_window(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_result: DecodedPredictionResult,
    sample_indices: IntVector,
    *,
    config: EvaluatorConfig,
) -> EvaluationSummary:
    decoded_offsets = require_decoded_offsets(decoded_result)
    aggregation = _aggregation(config)
    window_seconds = _required_int(config.window_seconds, "evaluation.window_seconds")
    repetitions = _required_int(config.repetitions, "evaluation.repetitions")
    seed = _required_int(config.seed, "evaluation.seed")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    chronological_samples = chronological_sample_view(store, sample_indices)
    first_timestamp = int(chronological_samples.sample_timestamps[0])
    last_timestamp = int(chronological_samples.sample_timestamps[-1])
    rng = np.random.default_rng(seed)
    runs = []
    if last_timestamp - first_timestamp <= window_seconds:
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                np.arange(sample_indices.shape[0], dtype=np.int64),
                aggregation=aggregation,
                metadata={"mode": "fullset_fallback"},
            )
        )
        return summarize_runs(runs, aggregation=aggregation)
    max_start = last_timestamp - window_seconds
    start_intervals = _non_empty_start_intervals(
        chronological_samples.sample_timestamps,
        first_timestamp=first_timestamp,
        max_start=max_start,
        window_seconds=window_seconds,
    )
    if not start_intervals:
        raise ValueError("uniform_window evaluation produced no non-empty windows")
    interval_sizes = np.array(
        [end - start + 1 for start, end in start_intervals],
        dtype=np.int64,
    )
    cumulative_sizes = np.cumsum(interval_sizes, dtype=np.int64)
    total_starts = int(cumulative_sizes[-1])
    for repetition in range(1, repetitions + 1):
        start_timestamp = _sample_start_timestamp(
            rng,
            start_intervals=start_intervals,
            cumulative_sizes=cumulative_sizes,
            total_starts=total_starts,
        )
        end_timestamp = start_timestamp + window_seconds
        selected_positions = chronological_samples.sample_positions[
            np.flatnonzero(
                (chronological_samples.sample_timestamps >= start_timestamp)
                & (chronological_samples.sample_timestamps < end_timestamp)
            )
        ].astype(np.int64, copy=False)
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                aggregation=aggregation,
                metadata={
                    "mode": "windowed",
                    "window_start_timestamp": start_timestamp,
                    "window_end_timestamp": end_timestamp,
                    "repetition": repetition,
                },
            )
        )
    return summarize_runs(runs, aggregation=aggregation)


def run_poisson_arrivals(
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_result: DecodedPredictionResult,
    sample_indices: IntVector,
    *,
    config: EvaluatorConfig,
) -> EvaluationSummary:
    decoded_offsets = require_decoded_offsets(decoded_result)
    aggregation = _aggregation(config)
    window_seconds = _required_int(config.window_seconds, "evaluation.window_seconds")
    repetitions = _required_int(config.repetitions, "evaluation.repetitions")
    seed = _required_int(config.seed, "evaluation.seed")
    arrival_rate = _required_float(
        config.arrival_rate_per_second,
        "evaluation.arrival_rate_per_second",
    )
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    chronological_samples = chronological_sample_view(store, sample_indices)
    first_timestamp = int(chronological_samples.sample_timestamps[0])
    last_timestamp = int(chronological_samples.sample_timestamps[-1])
    latest_start = last_timestamp - window_seconds
    if latest_start < first_timestamp:
        raise ValueError("Evaluation examples do not cover the requested replay window")

    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(repetitions):
        window_start = float(rng.uniform(first_timestamp, latest_start))
        window_end = window_start + window_seconds
        arrivals = sample_poisson_arrivals(
            rng,
            rate_per_second=arrival_rate,
            start_timestamp=window_start,
            end_timestamp=window_end,
        )
        selected_positions = chronological_samples.sample_positions[
            select_sample_positions_for_arrivals(
                chronological_samples.sample_timestamps,
                arrivals,
            )
        ].astype(np.int64, copy=False)
        if selected_positions.size == 0:
            continue
        runs.append(
            summarize_selected_costs(
                store,
                realization_policy,
                decoded_offsets,
                sample_indices,
                selected_positions,
                aggregation=aggregation,
                metadata={
                    "window_start_timestamp": window_start,
                    "window_end_timestamp": window_end,
                    "n_arrivals": int(arrivals.shape[0]),
                },
            )
        )

    if not runs:
        raise SpiceOperatorError(
            "poisson_arrivals evaluation produced no valid arrivals; "
            "adjust the benchmark rate or window"
        )
    return summarize_runs(runs, aggregation=aggregation)


def _non_empty_start_intervals(
    sample_timestamps: np.ndarray,
    *,
    first_timestamp: int,
    max_start: int,
    window_seconds: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for timestamp in sample_timestamps:
        start = max(first_timestamp, int(timestamp) - window_seconds + 1)
        end = min(max_start, int(timestamp))
        if start > end:
            continue
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def _sample_start_timestamp(
    rng: np.random.Generator,
    *,
    start_intervals: list[tuple[int, int]],
    cumulative_sizes: np.ndarray,
    total_starts: int,
) -> int:
    draw = int(rng.integers(total_starts))
    interval_index = int(np.searchsorted(cumulative_sizes, draw, side="right"))
    interval_start, _ = start_intervals[interval_index]
    interval_offset = draw - int(cumulative_sizes[interval_index - 1]) if interval_index else draw
    return interval_start + interval_offset


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field: {label}")
    return value


def _aggregation(config: EvaluatorConfig) -> EvaluationAggregation:
    return config.aggregation or EvaluationAggregation.TOTAL_RATIO


def _required_float(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required field: {label}")
    return value

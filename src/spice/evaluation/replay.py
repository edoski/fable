"""Replay evaluator engine and sampler policies."""

from __future__ import annotations

import numpy as np

from ..core.errors import SpiceOperatorError
from ..prediction.contracts import DecodedPredictionResult, require_decoded_offsets
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from .config import EvaluationAggregationId, EvaluatorConfig
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
    aggregation: EvaluationAggregationId,
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


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field: {label}")
    return value


def _aggregation(config: EvaluatorConfig) -> EvaluationAggregationId:
    if config.aggregation is None:
        raise ValueError("Missing required field: evaluation.aggregation")
    return config.aggregation.id


def _required_float(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required field: {label}")
    return value

"""Stochastic replay evaluator."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ...core.errors import SpiceOperatorError
from ...core.reporting import NullReporter, Reporter
from ...temporal.problem_store import CompiledProblemStore
from ..base import EvaluationSummary, EvaluatorConfig
from ..contracts import CompiledEvaluatorContract
from .shared import (
    EVALUATION_METRIC_DESCRIPTORS,
    sample_poisson_arrivals,
    select_sample_positions_for_arrivals,
    summarize_runs,
    summarize_selected_costs,
)

IntVector = NDArray[np.int64]


def _run(
    store: CompiledProblemStore,
    decoded_offsets: object,
    sample_indices: IntVector,
    reporter: Reporter | None,
    *,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
    seed: int,
) -> EvaluationSummary:
    reporter = reporter or NullReporter()
    if not isinstance(decoded_offsets, list):
        raise TypeError("poisson_replay decoded_offsets must be a list")
    if len(decoded_offsets) != int(sample_indices.shape[0]):
        raise ValueError("decoded_offsets must align with sample_indices")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    sample_timestamps = store.timestamps[store.anchor_rows[sample_indices]]
    first_timestamp = int(sample_timestamps[0])
    last_timestamp = int(sample_timestamps[-1])
    latest_start = last_timestamp - window_seconds
    if latest_start < first_timestamp:
        raise ValueError("Evaluation examples do not cover the requested replay window")

    rng = np.random.default_rng(seed)
    runs = []
    task_id = reporter.start_task(
        "evaluate replay",
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
        run = summarize_selected_costs(
            store,
            decoded_offsets,
            sample_indices,
            selected_positions,
            metadata={
                "window_start_timestamp": window_start,
                "window_end_timestamp": window_end,
                "n_arrivals": int(arrivals.shape[0]),
            },
        )
        runs.append(run)
        reporter.update_task(
            task_id,
            completed=repetition + 1,
            message=f"events={run.n_events}",
        )

    if not runs:
        raise SpiceOperatorError(
            "poisson_replay produced no valid arrivals; adjust the evaluator rate or window"
        )
    summary = summarize_runs(runs)
    reporter.finish_task(task_id, message=f"total_events={summary.total_events}")
    return summary


class PoissonReplayEvaluatorConfig(EvaluatorConfig[Literal["poisson_replay"]]):
    id: Literal["poisson_replay"] = "poisson_replay"
    window_seconds: int = Field(gt=0)
    arrival_rate_per_second: float = Field(gt=0.0)
    repetitions: int = Field(gt=0)
    seed: int = Field(ge=0)


def compile_evaluator(config: PoissonReplayEvaluatorConfig) -> CompiledEvaluatorContract:
    return CompiledEvaluatorContract(
        evaluator_id="poisson_replay",
        metric_descriptors=EVALUATION_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        config_payload=config.model_dump(mode="json", exclude_none=True),
        run_fn=lambda store, decoded_offsets, sample_indices, reporter: _run(
            store,
            decoded_offsets,
            sample_indices,
            reporter,
            window_seconds=config.window_seconds,
            arrival_rate_per_second=config.arrival_rate_per_second,
            repetitions=config.repetitions,
            seed=config.seed,
        ),
    )

"""Poisson replay evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import SpiceOperatorError
from .config import PoissonReplayEvaluatorConfig
from .contracts import CompiledEvaluatorContract
from .temporal_replay_runner import (
    TemporalReplaySampleView,
    TemporalReplaySelection,
    compile_temporal_replay_evaluator_contract,
)

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _ChronologicalSampleView:
    sample_positions: IntVector
    sample_timestamps: IntVector


def _sample_poisson_arrivals(
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


def _chronological_sample_view(samples: TemporalReplaySampleView) -> _ChronologicalSampleView:
    order = np.argsort(samples.sample_timestamps, kind="stable").astype(np.int64, copy=False)
    return _ChronologicalSampleView(
        sample_positions=samples.sample_positions[order],
        sample_timestamps=samples.sample_timestamps[order],
    )


def _select_sample_positions_for_arrivals(
    sample_timestamps: NDArray[np.int64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(sample_timestamps, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


class PoissonReplayAdapter:
    def __init__(self, config: PoissonReplayEvaluatorConfig) -> None:
        self.config = config

    def selections(
        self,
        samples: TemporalReplaySampleView,
    ) -> list[TemporalReplaySelection]:
        chronological_samples = _chronological_sample_view(samples)
        first_timestamp = int(chronological_samples.sample_timestamps[0])
        last_timestamp = int(chronological_samples.sample_timestamps[-1])
        latest_start = last_timestamp - self.config.window_seconds
        if latest_start < first_timestamp:
            raise ValueError("Evaluation examples do not cover the requested replay window")

        rng = np.random.default_rng(self.config.seed)
        selections = []
        for _ in range(self.config.repetitions):
            window_start = float(rng.uniform(first_timestamp, latest_start))
            window_end = window_start + self.config.window_seconds
            arrivals = _sample_poisson_arrivals(
                rng,
                rate_per_second=self.config.arrival_rate_per_second,
                start_timestamp=window_start,
                end_timestamp=window_end,
            )
            selected_positions = chronological_samples.sample_positions[
                _select_sample_positions_for_arrivals(
                    chronological_samples.sample_timestamps,
                    arrivals,
                )
            ].astype(np.int64, copy=False)
            if selected_positions.size == 0:
                continue
            selections.append(
                TemporalReplaySelection(
                    selected_positions=selected_positions,
                    metadata={
                        "window_start_timestamp": window_start,
                        "window_end_timestamp": window_end,
                        "n_arrivals": int(arrivals.shape[0]),
                    },
                )
            )
        return selections


def compile_poisson_replay_evaluator_contract(
    config: PoissonReplayEvaluatorConfig,
) -> CompiledEvaluatorContract:
    return compile_temporal_replay_evaluator_contract(
        evaluator_id=config.id,
        config=config,
        adapter=PoissonReplayAdapter(config),
        no_runs_error=_poisson_replay_no_runs_error(),
    )


def _poisson_replay_no_runs_error() -> SpiceOperatorError:
    return SpiceOperatorError(
        "poisson_arrivals evaluation produced no valid arrivals; "
        "adjust the benchmark rate or window"
    )

"""Poisson replay evaluator."""

from __future__ import annotations

import numpy as np

from ..core.errors import SpiceOperatorError
from ..temporal.problem_store import CompiledProblemStore
from .config import PoissonReplayEvaluatorConfig
from .contracts import CompiledEvaluatorContract, IntVector
from .sampling import (
    chronological_sample_view,
    sample_poisson_arrivals,
    select_sample_positions_for_arrivals,
)
from .temporal_replay_runner import (
    TemporalReplaySelection,
    compile_temporal_replay_evaluator_contract,
)


class PoissonReplayAdapter:
    def __init__(self, config: PoissonReplayEvaluatorConfig) -> None:
        self.config = config

    def selections(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> list[TemporalReplaySelection]:
        chronological_samples = chronological_sample_view(store, sample_indices)
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
            arrivals = sample_poisson_arrivals(
                rng,
                rate_per_second=self.config.arrival_rate_per_second,
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

    def no_runs_error(self) -> Exception:
        return SpiceOperatorError(
            "poisson_arrivals evaluation produced no valid arrivals; "
            "adjust the benchmark rate or window"
        )


def compile_poisson_replay_evaluator_contract(
    config: PoissonReplayEvaluatorConfig,
) -> CompiledEvaluatorContract:
    return compile_temporal_replay_evaluator_contract(
        evaluation_id=config.id,
        config_payload=config.model_dump(mode="json", exclude_none=True),
        adapter=PoissonReplayAdapter(config),
    )

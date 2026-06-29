"""Block-index Poisson replay evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import SpiceOperatorError
from .config import BlockPoissonReplayEvaluatorConfig
from .contracts import CompiledEvaluatorContract
from .temporal_replay_runner import (
    TemporalReplaySampleView,
    TemporalReplaySelection,
    compile_temporal_replay_evaluator_contract,
)

IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _ChronologicalBlockSampleView:
    sample_positions: IntVector
    sample_block_numbers: IntVector
    sample_timestamps: IntVector


def _sample_poisson_block_arrivals(
    rng: np.random.Generator,
    *,
    rate_per_block: float,
    start_block_offset: float,
    end_block_offset: float,
) -> NDArray[np.float64]:
    if rate_per_block <= 0:
        raise ValueError("rate_per_block must be positive")
    arrivals: list[float] = []
    cursor = start_block_offset
    while cursor < end_block_offset:
        cursor += rng.exponential(1.0 / rate_per_block)
        if cursor < end_block_offset:
            arrivals.append(cursor)
    return np.asarray(arrivals, dtype=np.float64)


def _chronological_block_sample_view(
    samples: TemporalReplaySampleView,
) -> _ChronologicalBlockSampleView:
    order = np.argsort(samples.sample_block_numbers, kind="stable").astype(
        np.int64,
        copy=False,
    )
    return _ChronologicalBlockSampleView(
        sample_positions=samples.sample_positions[order],
        sample_block_numbers=samples.sample_block_numbers[order],
        sample_timestamps=samples.sample_timestamps[order],
    )


def _select_sample_positions_for_block_arrivals(
    sample_block_offsets: NDArray[np.float64],
    arrivals: NDArray[np.float64],
) -> NDArray[np.int64]:
    if arrivals.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_positions = np.searchsorted(sample_block_offsets, arrivals, side="right") - 1
    return selected_positions[selected_positions >= 0].astype(np.int64, copy=False)


class BlockPoissonReplayAdapter:
    def __init__(self, config: BlockPoissonReplayEvaluatorConfig) -> None:
        self.config = config

    def selections(
        self,
        samples: TemporalReplaySampleView,
    ) -> list[TemporalReplaySelection]:
        chronological_samples = _chronological_block_sample_view(samples)
        sample_count = int(chronological_samples.sample_positions.shape[0])
        latest_start_offset = sample_count - self.config.window_blocks
        if latest_start_offset < 0:
            raise ValueError("Evaluation examples do not cover the requested block window")

        sample_block_offsets = np.arange(sample_count, dtype=np.float64)
        rng = np.random.default_rng(self.config.seed)
        selections = []
        for _ in range(self.config.repetitions):
            window_start_offset = int(rng.integers(0, latest_start_offset + 1))
            window_end_offset = window_start_offset + self.config.window_blocks
            arrivals = _sample_poisson_block_arrivals(
                rng,
                rate_per_block=self.config.arrival_rate_per_block,
                start_block_offset=float(window_start_offset),
                end_block_offset=float(window_end_offset),
            )
            selected_offsets = _select_sample_positions_for_block_arrivals(
                sample_block_offsets,
                arrivals,
            )
            in_window = selected_offsets[
                (selected_offsets >= window_start_offset)
                & (selected_offsets < window_end_offset)
            ]
            selected_positions = chronological_samples.sample_positions[in_window].astype(
                np.int64,
                copy=False,
            )
            if selected_positions.size == 0:
                continue
            first_offset = int(in_window.min())
            last_offset = int(in_window.max())
            selections.append(
                TemporalReplaySelection(
                    selected_positions=selected_positions,
                    metadata={
                        "window_start_block_offset": window_start_offset,
                        "window_end_block_offset": window_end_offset,
                        "window_blocks": self.config.window_blocks,
                        "window_start_block_number": int(
                            chronological_samples.sample_block_numbers[window_start_offset]
                        ),
                        "window_end_block_number_exclusive": int(
                            chronological_samples.sample_block_numbers[window_end_offset - 1]
                        )
                        + 1,
                        "window_first_timestamp": int(
                            chronological_samples.sample_timestamps[first_offset]
                        ),
                        "window_last_timestamp": int(
                            chronological_samples.sample_timestamps[last_offset]
                        ),
                        "n_arrivals": int(arrivals.shape[0]),
                    },
                )
            )
        return selections


def compile_block_poisson_replay_evaluator_contract(
    config: BlockPoissonReplayEvaluatorConfig,
) -> CompiledEvaluatorContract:
    return compile_temporal_replay_evaluator_contract(
        evaluator_id=config.id,
        config=config,
        adapter=BlockPoissonReplayAdapter(config),
        no_runs_error=_block_poisson_replay_no_runs_error(),
    )


def _block_poisson_replay_no_runs_error() -> SpiceOperatorError:
    return SpiceOperatorError(
        "block_poisson_arrivals evaluation produced no valid arrivals; "
        "adjust the benchmark rate or block window"
    )

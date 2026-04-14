"""Candidate-offset selection decode and replay helpers."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ....core.reporting import Reporter
from ....temporal.problem_store import CompiledProblemStore
from ...base import PredictionSimulationSummary
from ...offset_selection.replay import run_offset_replay
from .batch import CandidateSlateTargetBatch
from .outputs import candidate_logits, masked_candidate_logits

IntVector = NDArray[np.int64]


def allocate_prediction_buffer(sample_count: int) -> list[int]:
    return [0] * sample_count


def decode_into(
    predictions: object,
    sample_positions: torch.Tensor,
    outputs,
    targets: CandidateSlateTargetBatch,
) -> None:
    if not isinstance(predictions, list):
        raise TypeError("candidate_offset_selection prediction buffer must be a list")
    logits = masked_candidate_logits(candidate_logits(outputs), targets.candidate_mask)
    decoded = logits.argmax(dim=-1).cpu().tolist()
    positions = sample_positions.tolist()
    for sample_position, prediction in zip(positions, decoded, strict=True):
        predictions[int(sample_position)] = int(prediction)


def run_replay(
    store: CompiledProblemStore,
    predicted_offsets: object,
    sample_indices: IntVector,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
    seed: int,
    reporter: Reporter | None = None,
) -> PredictionSimulationSummary:
    return run_offset_replay(
        store,
        predicted_offsets,
        sample_indices,
        family_id="candidate_offset_selection",
        window_seconds=window_seconds,
        arrival_rate_per_second=arrival_rate_per_second,
        repetitions=repetitions,
        seed=seed,
        reporter=reporter,
    )

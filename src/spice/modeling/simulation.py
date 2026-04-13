"""Prediction-family replay helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.reporting import Reporter
from ..prediction import (
    CompiledPredictionContract,
    PredictionSimulationRun,
    PredictionSimulationSummary,
)
from ..temporal.problem_store import CompiledProblemStore

IntVector = NDArray[np.int64]
SimulationRunSummary = PredictionSimulationRun
SimulationSummary = PredictionSimulationSummary


def run_prediction_replay(
    prediction_contract: CompiledPredictionContract,
    store: CompiledProblemStore,
    predictions: object,
    *,
    sample_indices: IntVector,
    window_seconds: int,
    arrival_rate_per_second: float,
    repetitions: int,
    seed: int,
    reporter: Reporter | None = None,
) -> PredictionSimulationSummary:
    return prediction_contract.replay(
        store,
        predictions,
        sample_indices,
        window_seconds,
        arrival_rate_per_second,
        repetitions,
        seed,
        reporter,
    )

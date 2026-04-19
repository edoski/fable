"""Evaluator execution helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.reporting import Reporter
from ..evaluation import CompiledEvaluatorContract, EvaluationSummary
from ..prediction import DecodedOffsets
from ..temporal.problem_store import CompiledProblemStore

IntVector = NDArray[np.int64]


def run_prediction_evaluation(
    evaluator_contract: CompiledEvaluatorContract,
    store: CompiledProblemStore,
    decoded_offsets: DecodedOffsets,
    *,
    sample_indices: IntVector,
    reporter: Reporter | None = None,
) -> EvaluationSummary:
    return evaluator_contract.run(
        store,
        decoded_offsets,
        sample_indices,
        reporter,
    )

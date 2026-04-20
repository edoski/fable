"""Evaluator execution helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.reporting import Reporter
from ..evaluation import CompiledEvaluatorContract, EvaluationSummary
from ..prediction import DecodedOffsets
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract

IntVector = NDArray[np.int64]


def run_prediction_evaluation(
    evaluator_contract: CompiledEvaluatorContract,
    store: CompiledProblemStore,
    realization_policy: CompiledRealizationPolicyContract,
    decoded_offsets: DecodedOffsets,
    *,
    sample_indices: IntVector,
    reporter: Reporter | None = None,
) -> EvaluationSummary:
    return evaluator_contract.run(
        store,
        realization_policy,
        decoded_offsets,
        sample_indices,
        reporter,
    )

"""Model-to-evaluator scoring bridge."""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation import CompiledEvaluatorContract, EvaluationSummary
from ..prediction import CompiledPredictionContract
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from .inference import predict_with_model
from .models import TemporalModel
from .representations import CompiledRepresentationContract
from .runtime_planning import ModelingRuntimePlan


@dataclass(frozen=True, slots=True)
class EvaluationScoringRuntimePlan:
    model: TemporalModel
    prediction_contract: CompiledPredictionContract
    representation_contract: CompiledRepresentationContract
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector
    runtime_plan: ModelingRuntimePlan


def score_evaluation(
    *,
    scoring_plan: EvaluationScoringRuntimePlan,
    evaluator_contract: CompiledEvaluatorContract,
) -> EvaluationSummary:
    evaluator_contract.validate_prediction_contract(scoring_plan.prediction_contract)
    decoded_offsets = predict_with_model(
        scoring_plan.model,
        prediction_contract=scoring_plan.prediction_contract,
        representation_contract=scoring_plan.representation_contract,
        execution_policy=scoring_plan.execution_policy,
        store=scoring_plan.store,
        sample_indices=scoring_plan.sample_indices,
        runtime_plan=scoring_plan.runtime_plan,
    )
    return evaluator_contract.run(
        scoring_plan.store,
        scoring_plan.execution_policy,
        decoded_offsets,
        sample_indices=scoring_plan.sample_indices,
    )

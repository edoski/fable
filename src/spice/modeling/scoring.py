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
class ModelScoringInput:
    model: TemporalModel
    prediction_contract: CompiledPredictionContract
    representation_contract: CompiledRepresentationContract
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector
    runtime_plan: ModelingRuntimePlan


def score_evaluation(
    *,
    model_input: ModelScoringInput,
    evaluator_contract: CompiledEvaluatorContract,
) -> EvaluationSummary:
    evaluator_contract.validate_prediction_contract(model_input.prediction_contract)
    decoded_offsets = predict_with_model(
        model_input.model,
        prediction_contract=model_input.prediction_contract,
        representation_contract=model_input.representation_contract,
        execution_policy=model_input.execution_policy,
        store=model_input.store,
        sample_indices=model_input.sample_indices,
        runtime_plan=model_input.runtime_plan,
    )
    return evaluator_contract.run(
        model_input.store,
        model_input.execution_policy,
        decoded_offsets,
        sample_indices=model_input.sample_indices,
    )

"""Model-to-evaluator scoring bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..evaluation import CompiledEvaluatorContract, EvaluationSummary
from ..prediction import CompiledPredictionContract
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from .evaluation_runtime import (
    EvaluationScoringRuntimePlan,
    build_evaluation_scoring_runtime_plan,
)
from .families.base import ModelConfig
from .inference import predict_with_model
from .models import TemporalModel
from .representations import CompiledRepresentationContract


@dataclass(frozen=True, slots=True)
class ModelScoringInput:
    model: TemporalModel
    model_config: ModelConfig[Any]
    prediction_contract: CompiledPredictionContract
    representation_contract: CompiledRepresentationContract
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector
    runtime_plan: EvaluationScoringRuntimePlan


@dataclass(frozen=True, slots=True)
class EvaluationScoringContext:
    model_input: ModelScoringInput
    evaluator_contract: CompiledEvaluatorContract


def score_evaluation(context: EvaluationScoringContext) -> EvaluationSummary:
    model_input = context.model_input
    context.evaluator_contract.validate_prediction_contract(model_input.prediction_contract)
    decoded_offsets = predict_with_model(
        model_input.model,
        prediction_contract=model_input.prediction_contract,
        representation_contract=model_input.representation_contract,
        execution_policy=model_input.execution_policy,
        store=model_input.store,
        sample_indices=model_input.sample_indices,
        runtime_plan=model_input.runtime_plan,
    )
    return context.evaluator_contract.run(
        model_input.store,
        model_input.execution_policy,
        decoded_offsets,
        sample_indices=model_input.sample_indices,
    )


def build_model_scoring_input(
    *,
    model: TemporalModel,
    model_config: ModelConfig[Any],
    prediction_contract: CompiledPredictionContract,
    representation_contract: CompiledRepresentationContract,
    execution_policy: CompiledExecutionPolicyContract,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    runtime_plan: EvaluationScoringRuntimePlan | None = None,
    batch_size: int | None = None,
) -> ModelScoringInput:
    if runtime_plan is None:
        if batch_size is None:
            raise ValueError("batch_size is required when runtime_plan is not supplied")
        runtime_plan = build_evaluation_scoring_runtime_plan(
            model_config=model_config,
            batch_size=batch_size,
        )
    return ModelScoringInput(
        model=model,
        model_config=model_config,
        prediction_contract=prediction_contract,
        representation_contract=representation_contract,
        execution_policy=execution_policy,
        store=store,
        sample_indices=sample_indices,
        runtime_plan=runtime_plan,
    )


def build_evaluation_scoring_context(
    *,
    model_input: ModelScoringInput,
    evaluator_contract: CompiledEvaluatorContract,
) -> EvaluationScoringContext:
    return EvaluationScoringContext(
        model_input=model_input,
        evaluator_contract=evaluator_contract,
    )

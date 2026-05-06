# pyright: strict

"""Model-bound objective runtime production."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.errors import ConfigResolutionError
from ..evaluation import CompiledEvaluatorContract
from ..metrics import MetricDescriptor, MetricSet
from ..objectives import CompiledObjectiveContract, ObjectiveConfig, compile_objective_contract
from ..prediction import CompiledPredictionContract
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from .models import TemporalModel
from .representations import CompiledRepresentationContract
from .runtime_planning import ModelingRuntimePlan
from .scoring import EvaluationScoringRuntimePlan, score_evaluation


@dataclass(frozen=True, slots=True)
class ObjectiveMetricContext:
    model: TemporalModel
    prediction_contract: CompiledPredictionContract
    representation_contract: CompiledRepresentationContract
    execution_policy: CompiledExecutionPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector
    runtime_plan: ModelingRuntimePlan


EvaluateObjectiveMetricsFn = Callable[
    [MetricSet, ObjectiveMetricContext | None],
    MetricSet,
]


@dataclass(frozen=True, slots=True)
class CompiledObjectiveRuntime:
    contract: CompiledObjectiveContract
    evaluate_metrics_fn: EvaluateObjectiveMetricsFn

    def evaluate_metrics(
        self,
        validation_metrics: MetricSet,
        *,
        context: ObjectiveMetricContext | None = None,
    ) -> MetricSet:
        return self.evaluate_metrics_fn(validation_metrics, context)


def compile_objective_runtime(
    objective: ObjectiveConfig,
    *,
    evaluator_contract: CompiledEvaluatorContract | None,
    prediction_metric_descriptors: tuple[MetricDescriptor, ...],
) -> CompiledObjectiveRuntime:
    contract = compile_objective_contract(
        objective,
        evaluator_id=None if evaluator_contract is None else evaluator_contract.evaluator_id,
    )
    if contract.objective_id == "validation":
        _require_metric_descriptor(
            contract.metric_id,
            prediction_metric_descriptors,
            owner="prediction",
        )
        return CompiledObjectiveRuntime(
            contract=contract,
            evaluate_metrics_fn=_validation_metrics,
        )
    if evaluator_contract is None:
        raise ConfigResolutionError("evaluation objective runtime requires evaluation config")
    descriptor = _require_metric_descriptor(
        contract.metric_id,
        evaluator_contract.metric_descriptors,
        owner="evaluator",
    )
    if descriptor.direction is not None and descriptor.direction != contract.direction:
        raise ConfigResolutionError(
            "objective direction does not match evaluator metric direction: "
            f"{contract.metric_id} is {descriptor.direction}, objective is "
            f"{contract.direction}"
        )

    def _evaluate(
        validation_metrics: MetricSet,
        context: ObjectiveMetricContext | None,
    ) -> MetricSet:
        del validation_metrics
        if context is None:
            raise ValueError("evaluation objective runtime requires objective metric context")
        return score_evaluation(
            scoring_plan=EvaluationScoringRuntimePlan(
                model=context.model,
                prediction_contract=context.prediction_contract,
                representation_contract=context.representation_contract,
                execution_policy=context.execution_policy,
                store=context.store,
                sample_indices=context.sample_indices,
                runtime_plan=context.runtime_plan,
            ),
            evaluator_contract=evaluator_contract,
        ).metrics

    return CompiledObjectiveRuntime(contract=contract, evaluate_metrics_fn=_evaluate)


def _validation_metrics(
    validation_metrics: MetricSet,
    context: ObjectiveMetricContext | None,
) -> MetricSet:
    del context
    return validation_metrics


def _require_metric_descriptor(
    metric_id: str,
    descriptors: tuple[MetricDescriptor, ...],
    *,
    owner: str,
) -> MetricDescriptor:
    for descriptor in descriptors:
        if descriptor.id == metric_id:
            return descriptor
    known = ", ".join(sorted(descriptor.id for descriptor in descriptors))
    raise ConfigResolutionError(
        f"objective metric {metric_id} is not declared by {owner} metrics. Known metrics: {known}"
    )

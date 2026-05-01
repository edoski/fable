# pyright: strict

"""Model-bound objective runtime production."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.errors import ConfigResolutionError
from ..evaluation import EvaluatorConfig, compile_evaluator_contract
from ..objectives import CompiledObjectiveContract, ObjectiveConfig, compile_objective_contract
from ..prediction import MetricDescriptor, MetricSet
from .scoring import (
    ModelScoringInput,
    build_evaluation_scoring_context,
    score_evaluation,
)

EvaluateObjectiveMetricsFn = Callable[
    [MetricSet, ModelScoringInput],
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
        context: ModelScoringInput,
    ) -> MetricSet:
        return self.evaluate_metrics_fn(validation_metrics, context)


def compile_objective_runtime(
    objective: ObjectiveConfig,
    *,
    evaluation: EvaluatorConfig | None,
    prediction_metric_descriptors: tuple[MetricDescriptor, ...],
) -> CompiledObjectiveRuntime:
    contract = compile_objective_contract(objective, evaluation=evaluation)
    if contract.objective_id == "validation":
        _require_metric_descriptor(
            contract.metric_id,
            prediction_metric_descriptors,
            owner="prediction",
        )
        return CompiledObjectiveRuntime(
            contract=contract,
            evaluate_metrics_fn=lambda validation_metrics, context: validation_metrics,
        )
    evaluator_contract = compile_evaluator_contract(_require_evaluation(contract, evaluation))
    _require_metric_descriptor(
        contract.metric_id,
        evaluator_contract.metric_descriptors,
        owner="evaluator",
    )

    def _evaluate(
        validation_metrics: MetricSet,
        context: ModelScoringInput,
    ) -> MetricSet:
        del validation_metrics
        return score_evaluation(
            build_evaluation_scoring_context(
                model_input=context,
                evaluator_contract=evaluator_contract,
            )
        ).metrics

    return CompiledObjectiveRuntime(contract=contract, evaluate_metrics_fn=_evaluate)


def _require_metric_descriptor(
    metric_id: str,
    descriptors: tuple[MetricDescriptor, ...],
    *,
    owner: str,
) -> None:
    if metric_id in {descriptor.id for descriptor in descriptors}:
        return
    known = ", ".join(sorted(descriptor.id for descriptor in descriptors))
    raise ConfigResolutionError(
        f"objective metric {metric_id} is not declared by {owner} metrics. Known metrics: {known}"
    )


def _require_evaluation(
    contract: CompiledObjectiveContract,
    evaluation: EvaluatorConfig | None,
) -> EvaluatorConfig:
    if evaluation is None:
        raise TypeError("evaluation objective runtime requires evaluation config")
    if contract.benchmark_id != evaluation.id:
        raise TypeError(
            f"objective benchmark {contract.benchmark_id} does not match evaluation {evaluation.id}"
        )
    return evaluation

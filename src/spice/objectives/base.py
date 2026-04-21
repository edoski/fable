"""Thin workflow-owned objective policy seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import field_validator, model_validator

from ..core.errors import ConfigResolutionError
from ..core.validation import validate_path_segment
from ..evaluation import EvaluatorConfig, compile_evaluator_contract
from ..modeling.families.base import ConfigModel
from ..prediction import MetricSet
from ..semantics import ObjectiveSemantics

if TYPE_CHECKING:
    from ..modeling.families.base import ModelConfig
    from ..modeling.models import TemporalModel
    from ..modeling.representations import CompiledRepresentationContract
    from ..prediction import CompiledPredictionContract
    from ..temporal.problem_store import CompiledProblemStore, IntVector
    from ..temporal.realization import CompiledRealizationPolicyContract


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveConfig(ConfigModel):
    id: str
    metric_id: str
    direction: ObjectiveDirection
    benchmark_id: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="objective.id")

    @field_validator("metric_id")
    @classmethod
    def validate_metric_id(cls, value: str) -> str:
        return validate_path_segment(value, label="objective.metric_id")

    @field_validator("benchmark_id")
    @classmethod
    def validate_benchmark_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_path_segment(value, label="objective.benchmark_id")

    @model_validator(mode="after")
    def validate_shape(self) -> ObjectiveConfig:
        if self.id == "validation":
            if self.benchmark_id is not None:
                raise ValueError("validation objectives must not declare benchmark_id")
            return self
        if self.id == "evaluation":
            if self.benchmark_id is None:
                raise ValueError("evaluation objectives must declare benchmark_id")
            return self
        raise ValueError("objective.id must be one of: validation, evaluation")


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluationContext:
    model: TemporalModel
    model_config: ModelConfig
    prediction_contract: CompiledPredictionContract
    representation_contract: CompiledRepresentationContract
    realization_policy: CompiledRealizationPolicyContract
    store: CompiledProblemStore
    sample_indices: IntVector
    batch_size: int


EvaluateObjectiveMetricsFn = Callable[[MetricSet, ObjectiveEvaluationContext], MetricSet]


@dataclass(frozen=True, slots=True)
class CompiledObjectiveContract:
    objective_id: str
    metric_id: str
    direction: Literal["maximize", "minimize"]
    benchmark_id: str | None
    config_payload: dict[str, object]
    evaluate_metrics_fn: EvaluateObjectiveMetricsFn

    @property
    def semantics(self) -> ObjectiveSemantics:
        return ObjectiveSemantics(
            objective_id=self.objective_id,
            metric_id=self.metric_id,
            direction=self.direction,
            benchmark_id=self.benchmark_id,
        )

    @property
    def checkpoint_monitor(self) -> str:
        if self.benchmark_id is None:
            return f"validation_{self.metric_id}"
        return f"validation_{self.benchmark_id}_{self.metric_id}"

    def evaluate_metrics(
        self,
        validation_metrics: MetricSet,
        *,
        context: ObjectiveEvaluationContext,
    ) -> MetricSet:
        return self.evaluate_metrics_fn(validation_metrics, context)

    def value(self, metrics: MetricSet) -> float:
        return metrics.require(self.metric_id)


def coerce_objective_config(
    payload: Mapping[str, object] | ObjectiveConfig,
) -> ObjectiveConfig:
    if isinstance(payload, ObjectiveConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError("objective must be a mapping")
    return ObjectiveConfig.model_validate(dict(payload))


def compile_objective_contract(
    config: ObjectiveConfig,
    *,
    evaluation: EvaluatorConfig | None,
) -> CompiledObjectiveContract:
    payload = config.model_dump(mode="json", exclude_none=True)
    if config.id == "validation":
        return CompiledObjectiveContract(
            objective_id="validation",
            metric_id=config.metric_id,
            direction=config.direction.value,
            benchmark_id=None,
            config_payload=payload,
            evaluate_metrics_fn=lambda validation_metrics, context: validation_metrics,
        )
    if evaluation is None:
        raise ConfigResolutionError(
            f"objective benchmark {config.benchmark_id} requires evaluation"
        )
    if config.benchmark_id != evaluation.id:
        raise ConfigResolutionError(
            f"objective benchmark {config.benchmark_id} does not match evaluation {evaluation.id}"
        )
    evaluator_contract = compile_evaluator_contract(evaluation)

    def _evaluate(validation_metrics: MetricSet, context: ObjectiveEvaluationContext) -> MetricSet:
        del validation_metrics
        from ..modeling.inference import predict_with_model

        decoded_offsets = predict_with_model(
            context.model,
            model_config=context.model_config,
            prediction_contract=context.prediction_contract,
            representation_contract=context.representation_contract,
            store=context.store,
            sample_indices=context.sample_indices,
            batch_size=context.batch_size,
        )
        return evaluator_contract.run(
            context.store,
            context.realization_policy,
            decoded_offsets,
            context.sample_indices,
        ).metrics

    return CompiledObjectiveContract(
        objective_id="evaluation",
        metric_id=config.metric_id,
        direction=config.direction.value,
        benchmark_id=evaluator_contract.evaluation_id,
        config_payload=payload,
        evaluate_metrics_fn=_evaluate,
    )

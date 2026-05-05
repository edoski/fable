"""Thin workflow-owned objective policy seam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from ..core.specs import owner_payload, validate_owner_config
from ..core.validation import validate_path_segment
from ..metrics import MetricSet
from ..semantics import ObjectiveSemantics


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
class CompiledObjectiveContract:
    objective_id: str
    metric_id: str
    direction: Literal["maximize", "minimize"]
    benchmark_id: str | None

    @property
    def semantics(self) -> ObjectiveSemantics:
        return ObjectiveSemantics(
            objective_id=self.objective_id,
            metric_id=self.metric_id,
            direction=self.direction,
            benchmark_id=self.benchmark_id,
        )

    def value(self, metrics: MetricSet) -> float:
        return metrics.require(self.metric_id)


def coerce_objective_config(
    payload: object,
) -> ObjectiveConfig:
    if isinstance(payload, ObjectiveConfig):
        return payload
    return validate_owner_config(
        owner_payload(payload, owner="objective", config_type=ObjectiveConfig),
        ObjectiveConfig,
    )


def compile_objective_contract(
    config: ObjectiveConfig,
    *,
    evaluator_id: str | None,
) -> CompiledObjectiveContract:
    if config.id == "validation":
        return CompiledObjectiveContract(
            objective_id="validation",
            metric_id=config.metric_id,
            direction=config.direction.value,
            benchmark_id=None,
        )
    if evaluator_id is None:
        raise ConfigResolutionError(
            f"objective benchmark {config.benchmark_id} requires evaluation"
        )
    if config.benchmark_id != evaluator_id:
        raise ConfigResolutionError(
            f"objective benchmark {config.benchmark_id} does not match evaluator {evaluator_id}"
        )
    return CompiledObjectiveContract(
        objective_id="evaluation",
        metric_id=config.metric_id,
        direction=config.direction.value,
        benchmark_id=evaluator_id,
    )

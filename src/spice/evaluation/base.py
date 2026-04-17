"""Evaluator config and runtime result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ..prediction.base import MetricDescriptor, MetricSet, WindowMetricSummary


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


class EvaluationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvaluatorConfig(EvaluationConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="evaluation.evaluator.id")


EvaluationMetadataValue = str | int | float


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, EvaluationMetadataValue]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[EvaluationRun]


@dataclass(frozen=True, slots=True)
class EvaluatorSemantics:
    evaluator_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]

"""Shared prediction-family types and generic metric/output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


class PredictionConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PredictionFamilyConfig(PredictionConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="prediction.family.id")


@dataclass(frozen=True, slots=True)
class MetricDescriptor:
    id: str
    label: str
    role: Literal["primary", "secondary", "diagnostic"]


@dataclass(frozen=True, slots=True)
class MetricSet:
    values: dict[str, float]

    def require(self, metric_id: str) -> float:
        try:
            return self.values[metric_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.values))
            raise ValueError(f"Unknown metric id: {metric_id}. Known metrics: {known}") from exc


@dataclass(frozen=True, slots=True)
class WindowMetricSummary:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class PredictionHeadSpec:
    id: str
    size: int


@dataclass(frozen=True, slots=True)
class PredictionOutputSpec:
    heads: tuple[PredictionHeadSpec, ...]

    def head(self, head_id: str) -> PredictionHeadSpec:
        for head in self.heads:
            if head.id == head_id:
                return head
        known = ", ".join(head.id for head in self.heads) or "<none>"
        raise ValueError(f"Unknown output head: {head_id}. Known heads: {known}")

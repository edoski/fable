"""Evaluation config models."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.validation import validate_path_segment


class EvaluationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvaluationSampler(StrEnum):
    FULLSET = "fullset"
    POISSON_ARRIVALS = "poisson_arrivals"


class EvaluationAggregationId(StrEnum):
    EVENT_MEAN = "event_mean"
    TOTAL_RATIO = "total_ratio"


class EvaluationAggregationConfig(EvaluationConfigModel):
    id: EvaluationAggregationId


class EvaluationEngine(StrEnum):
    REPLAY = "replay"
    ZERO_STOP_ROLLOUT = "zero_stop_rollout"
    ANCHOR_BASEFEE = "anchor_basefee"


class EvaluatorConfig(EvaluationConfigModel):
    id: str
    engine: EvaluationEngine = EvaluationEngine.REPLAY
    sampler: EvaluationSampler | None = None
    window_seconds: int | None = Field(default=None, gt=0)
    repetitions: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0)
    arrival_rate_per_second: float | None = Field(default=None, gt=0.0)
    aggregation: EvaluationAggregationConfig | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="evaluation.id")

    @model_validator(mode="after")
    def validate_sampler_fields(self) -> Self:
        if self.engine is not EvaluationEngine.REPLAY:
            _require_absent(
                self.sampler,
                self.window_seconds,
                self.repetitions,
                self.seed,
                self.arrival_rate_per_second,
                self.aggregation,
                labels=(
                    "evaluation.sampler",
                    "evaluation.window_seconds",
                    "evaluation.repetitions",
                    "evaluation.seed",
                    "evaluation.arrival_rate_per_second",
                    "evaluation.aggregation",
                ),
            )
            return self
        if self.sampler is None:
            raise ValueError("Missing required fields: evaluation.sampler")
        _require_present(
            self.aggregation,
            labels=("evaluation.aggregation",),
        )
        if self.sampler is EvaluationSampler.FULLSET:
            _require_absent(
                self.window_seconds,
                self.repetitions,
                self.seed,
                self.arrival_rate_per_second,
                labels=(
                    "evaluation.window_seconds",
                    "evaluation.repetitions",
                    "evaluation.seed",
                    "evaluation.arrival_rate_per_second",
                ),
            )
            return self
        _require_present(
            self.window_seconds,
            self.repetitions,
            self.seed,
            self.arrival_rate_per_second,
            labels=(
                "evaluation.window_seconds",
                "evaluation.repetitions",
                "evaluation.seed",
                "evaluation.arrival_rate_per_second",
            ),
        )
        return self


def _require_present(*values: object, labels: tuple[str, ...]) -> None:
    missing = [label for label, value in zip(labels, values, strict=True) if value is None]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def _require_absent(*values: object, labels: tuple[str, ...]) -> None:
    unexpected = [label for label, value in zip(labels, values, strict=True) if value is not None]
    if unexpected:
        raise ValueError("Unexpected fields for evaluation sampler: " + ", ".join(unexpected))

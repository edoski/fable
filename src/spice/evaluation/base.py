"""Evaluator config and runtime result contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from ..core.closed_dispatch import validate_path_segment
from ..core.reporting import Reporter
from ..prediction.base import MetricDescriptor, MetricSet, WindowMetricSummary
from ..prediction.contracts import DecodedOffsets
from ..temporal.problem_store import CompiledProblemStore


class EvaluationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvaluatorConfig(EvaluationConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_path_segment(value, label="evaluation.evaluator.id")


EvaluationMetadataValue = str | int | float
IntVector = NDArray[np.int64]


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


RunEvaluatorFn = Callable[
    [CompiledProblemStore, DecodedOffsets, IntVector, Reporter | None],
    EvaluationSummary,
]


@dataclass(frozen=True, slots=True)
class CompiledEvaluatorContract:
    evaluator_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    config_payload: dict[str, object]
    run_fn: RunEvaluatorFn

    def run(
        self,
        store: CompiledProblemStore,
        decoded_offsets: DecodedOffsets,
        sample_indices: IntVector,
        reporter: Reporter | None,
    ) -> EvaluationSummary:
        return self.run_fn(store, decoded_offsets, sample_indices, reporter)

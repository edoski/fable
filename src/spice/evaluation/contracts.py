"""Compiled evaluator contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.reporting import Reporter
from ..prediction.base import MetricDescriptor
from ..temporal.problem_store import CompiledProblemStore
from .base import EvaluationSummary, EvaluatorSemantics

IntVector = NDArray[np.int64]
RunEvaluatorFn = Callable[
    [CompiledProblemStore, object, IntVector, Reporter | None],
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

    @property
    def semantics(self) -> EvaluatorSemantics:
        return EvaluatorSemantics(
            evaluator_id=self.evaluator_id,
            metric_descriptors=self.metric_descriptors,
            primary_metric_id=self.primary_metric_id,
            direction=self.direction,
        )

    def run(
        self,
        store: CompiledProblemStore,
        decoded_offsets: object,
        sample_indices: IntVector,
        reporter: Reporter | None,
    ) -> EvaluationSummary:
        return self.run_fn(store, decoded_offsets, sample_indices, reporter)

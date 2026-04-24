"""Compiled evaluator contracts and summary models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ..prediction.base import MetricDescriptor, MetricSet, WindowMetricSummary
from ..prediction.contracts import DecodedPredictionResult
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract

if TYPE_CHECKING:
    from ..prediction import CompiledPredictionContract

EvaluationMetadataValue = str | int | float
IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    n_events: int
    metrics: dict[str, float]
    metadata: dict[str, EvaluationMetadataValue]
    event_metric_sums: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    metrics: MetricSet
    window_metrics: dict[str, WindowMetricSummary]
    total_events: int
    runs: list[EvaluationRun]


RunEvaluatorFn = Callable[
    [
        CompiledProblemStore,
        CompiledRealizationPolicyContract,
        DecodedPredictionResult,
        IntVector,
    ],
    EvaluationSummary,
]


@dataclass(frozen=True, slots=True)
class CompiledEvaluatorContract:
    evaluation_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    config_payload: dict[str, object]
    accepted_decoded_result_id: str
    run_fn: RunEvaluatorFn

    def run(
        self,
        store: CompiledProblemStore,
        realization_policy: CompiledRealizationPolicyContract,
        decoded_result: DecodedPredictionResult,
        sample_indices: IntVector,
    ) -> EvaluationSummary:
        if decoded_result.decoded_result_id != self.accepted_decoded_result_id:
            raise TypeError(
                "Evaluator decoded-result requirement does not match prediction output: "
                f"{self.accepted_decoded_result_id} != {decoded_result.decoded_result_id}"
            )
        return self.run_fn(
            store,
            realization_policy,
            decoded_result,
            sample_indices,
        )

    def validate_prediction_contract(
        self,
        prediction_contract: CompiledPredictionContract,
    ) -> None:
        if prediction_contract.decoded_result_id != self.accepted_decoded_result_id:
            raise TypeError(
                "Evaluator decoded-result requirement does not match prediction contract: "
                f"{self.accepted_decoded_result_id} != {prediction_contract.decoded_result_id}"
            )

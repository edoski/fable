"""Compiled evaluator contracts and summary models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..metrics import MetricDescriptor, MetricSet, WindowMetricSummary
from ..prediction.decoding import DecodedPredictionResult
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from .config import EvaluatorConfig

if TYPE_CHECKING:
    from ..prediction import CompiledPredictionContract

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
    [
        CompiledProblemStore,
        CompiledExecutionPolicyContract,
        DecodedPredictionResult,
        PreparedActionSpace,
    ],
    EvaluationSummary,
]


@dataclass(frozen=True, slots=True)
class CompiledEvaluatorContract:
    evaluator_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    config: EvaluatorConfig
    accepted_decoded_result_id: str
    run_fn: RunEvaluatorFn

    def __post_init__(self) -> None:
        descriptor_ids = [descriptor.id for descriptor in self.metric_descriptors]
        if len(descriptor_ids) != len(set(descriptor_ids)):
            raise ValueError("Evaluator metric descriptor ids must be unique")
        primary_descriptors = [
            descriptor for descriptor in self.metric_descriptors if descriptor.role == "primary"
        ]
        if len(primary_descriptors) != 1:
            raise ValueError("Evaluator contract must declare exactly one primary metric")

    @property
    def primary_metric_descriptor(self) -> MetricDescriptor:
        return next(
            descriptor
            for descriptor in self.metric_descriptors
            if descriptor.role == "primary"
        )

    @property
    def primary_metric_id(self) -> str:
        return self.primary_metric_descriptor.id

    def run(
        self,
        store: CompiledProblemStore,
        execution_policy: CompiledExecutionPolicyContract,
        decoded_result: DecodedPredictionResult,
        action_space: PreparedActionSpace,
    ) -> EvaluationSummary:
        if decoded_result.decoded_result_id != self.accepted_decoded_result_id:
            raise TypeError(
                "Evaluator decoded-result requirement does not match prediction output: "
                f"{self.accepted_decoded_result_id} != {decoded_result.decoded_result_id}"
            )
        return self.run_fn(
            store,
            execution_policy,
            decoded_result,
            action_space,
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

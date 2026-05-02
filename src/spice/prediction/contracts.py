"""Compiled prediction contracts and batch protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..metrics import MetricDescriptor, MetricSet
from ..semantics import PredictionSemantics
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from .base import PredictionOutputSpec
from .decoding import ActionSpaceDecodeContext, DecodedPredictionResult

if TYPE_CHECKING:
    from ..modeling.models import ModelOutputs

IntVector = NDArray[np.int64]


class ModelInputBatch(Protocol):
    @property
    def sample_positions(self) -> torch.Tensor: ...

    @property
    def action_mask(self) -> torch.Tensor: ...

    def to_device(self, device: torch.device) -> ModelInputBatch: ...

    def model_kwargs(self) -> Mapping[str, torch.Tensor]: ...

    def pin_memory(self) -> ModelInputBatch: ...


class PredictionTargetBatch(Protocol):
    def to_device(self, device: torch.device) -> PredictionTargetBatch: ...

    def pin_memory(self) -> PredictionTargetBatch: ...


class PreparedPredictionTargets(Protocol):
    @property
    def estimated_storage_bytes(self) -> int: ...

    def build_batch(self, sample_positions: torch.Tensor) -> PredictionTargetBatch: ...

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedPredictionTargets: ...


class EpochMetricAccumulator(Protocol):
    def update(self, batch_state: object) -> None: ...

    def snapshot(self) -> MetricSet: ...

    def finalize(self) -> MetricSet: ...


BuildOutputSpecFn = Callable[[int], PredictionOutputSpec]
FitTrainingStateFn = Callable[
    [CompiledProblemStore, IntVector, CompiledExecutionPolicyContract],
    object | None,
]
# Training state is reusable semantic state. Implementations may cache
# device/dtype views during loss computation, but semantic values must not
# mutate or depend on batch call order.
PrepareTargetsFn = Callable[
    [CompiledProblemStore, CompiledExecutionPolicyContract, PreparedActionSpace],
    PreparedPredictionTargets,
]
ComputeBatchLossAndStateFn = Callable[
    [Any, PredictionTargetBatch, object | None],
    tuple[torch.Tensor, object],
]
CreateEpochAccumulatorFn = Callable[[], EpochMetricAccumulator]
AllocateDecodedResultFn = Callable[[int], DecodedPredictionResult]
DecodeBatchResultIntoFn = Callable[
    [DecodedPredictionResult, Any, ActionSpaceDecodeContext],
    None,
]


@dataclass(slots=True)
class PredictionBatch:
    inputs: ModelInputBatch
    targets: PredictionTargetBatch

    @property
    def sample_positions(self) -> torch.Tensor:
        return self.inputs.sample_positions

    def to_device(self, device: torch.device) -> PredictionBatch:
        inputs = self.inputs.to_device(device)
        targets = self.targets.to_device(device)
        if inputs is self.inputs and targets is self.targets:
            return self
        return PredictionBatch(inputs=inputs, targets=targets)

    def model_kwargs(self) -> Mapping[str, torch.Tensor]:
        return self.inputs.model_kwargs()

    def pin_memory(self) -> PredictionBatch:
        return PredictionBatch(
            inputs=self.inputs.pin_memory(),
            targets=self.targets.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class CompiledPredictionContract:
    prediction_id: str
    prediction_family_id: str
    training_metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    build_output_spec_fn: BuildOutputSpecFn
    prepare_targets_fn: PrepareTargetsFn
    compute_batch_loss_and_state_fn: ComputeBatchLossAndStateFn
    create_epoch_accumulator_fn: CreateEpochAccumulatorFn
    decoded_result_id: str
    allocate_decoded_result_fn: AllocateDecodedResultFn
    decode_batch_result_into_fn: DecodeBatchResultIntoFn
    fit_training_state_fn: FitTrainingStateFn | None = None

    @property
    def semantics(self) -> PredictionSemantics:
        return PredictionSemantics(
            prediction_id=self.prediction_id,
            prediction_family_id=self.prediction_family_id,
            training_metric_descriptors=self.training_metric_descriptors,
            primary_metric_id=self.primary_metric_id,
            direction=self.direction,
        )

    def build_output_spec(self, max_candidate_slots: int) -> PredictionOutputSpec:
        return self.build_output_spec_fn(max_candidate_slots)

    def fit_training_state(
        self,
        store: CompiledProblemStore,
        train_sample_indices: IntVector,
        *,
        execution_policy: CompiledExecutionPolicyContract,
    ) -> object | None:
        if self.fit_training_state_fn is None:
            return None
        return self.fit_training_state_fn(store, train_sample_indices, execution_policy)

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        *,
        execution_policy: CompiledExecutionPolicyContract,
        action_space: PreparedActionSpace,
    ) -> PreparedPredictionTargets:
        return self.prepare_targets_fn(store, execution_policy, action_space)

    def compute_batch_loss_and_state(
        self,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
        *,
        training_state: object | None,
    ) -> tuple[torch.Tensor, object]:
        return self.compute_batch_loss_and_state_fn(outputs, targets, training_state)

    def create_epoch_accumulator(self) -> EpochMetricAccumulator:
        return self.create_epoch_accumulator_fn()

    def allocate_decoded_result(self, sample_count: int) -> DecodedPredictionResult:
        return self.allocate_decoded_result_fn(sample_count)

    def decode_batch_result_into(
        self,
        predictions: DecodedPredictionResult,
        outputs: ModelOutputs,
        decode_context: ActionSpaceDecodeContext,
    ) -> None:
        if predictions.decoded_result_id != self.decoded_result_id:
            raise TypeError(
                "Prediction decoded result kind does not match prediction contract: "
                f"{predictions.decoded_result_id} != {self.decoded_result_id}"
            )
        self.decode_batch_result_into_fn(predictions, outputs, decode_context)

"""Compiled prediction contracts and batch wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.reporting import StageMetricDescriptor, StageMetricValue, format_compact_number
from ..semantics import PredictionSemantics
from ..temporal.problem_store import CompiledProblemStore
from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionOutputSpec,
)

if TYPE_CHECKING:
    from ..modeling.models import ModelOutputs
    from ..modeling.representations import PreparedRepresentation

IntVector = NDArray[np.int64]


class ModelInputBatch(Protocol):
    @property
    def sample_positions(self) -> torch.Tensor: ...

    def to_device(self, device: torch.device) -> ModelInputBatch: ...

    def model_kwargs(self) -> Mapping[str, torch.Tensor]: ...

    def pin_memory(self) -> ModelInputBatch: ...


class PredictionTargetBatch(Protocol):
    def to_device(self, device: torch.device) -> PredictionTargetBatch: ...

    def pin_memory(self) -> PredictionTargetBatch: ...


class PreparedPredictionTargets(Protocol):
    @property
    def storage_mode_id(self) -> str: ...

    @property
    def estimated_storage_bytes(self) -> int: ...

    def build_batch(self, sample_positions: torch.Tensor) -> PredictionTargetBatch: ...

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedPredictionTargets | None: ...


class EpochMetricAccumulator(Protocol):
    def update(self, batch_state: object) -> None: ...

    def snapshot(self) -> MetricSet: ...

    def finalize(self) -> MetricSet: ...


BuildOutputSpecFn = Callable[[int], PredictionOutputSpec]
FitTrainingStateFn = Callable[[CompiledProblemStore, IntVector], object | None]
PrepareTargetsFn = Callable[[CompiledProblemStore, IntVector], PreparedPredictionTargets]
ComputeBatchLossAndStateFn = Callable[
    [Any, PredictionTargetBatch, object | None],
    tuple[torch.Tensor, object],
]
CreateEpochAccumulatorFn = Callable[[str], EpochMetricAccumulator]
SelectBestEpochFn = Callable[[list[MetricSet]], int]
AllocateDecodedOffsetsFn = Callable[[int], object]
DecodeSelectedOffsetsIntoFn = Callable[
    [object, torch.Tensor, Any, PredictionTargetBatch],
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


@dataclass(slots=True)
class PredictionPreparedRepresentation:
    prepared: PreparedRepresentation
    targets: PreparedPredictionTargets

    @property
    def representation_id(self) -> str:
        return self.prepared.representation_id

    @property
    def input_storage_mode_id(self) -> str:
        return self.prepared.storage_mode_id

    @property
    def target_storage_mode_id(self) -> str:
        return self.targets.storage_mode_id

    @property
    def batch_planner_id(self) -> str:
        return self.prepared.batch_planner_id

    @property
    def sample_count(self) -> int:
        return self.prepared.sample_count

    @property
    def batch_signatures(self) -> IntVector:
        return self.prepared.batch_signatures

    @property
    def estimated_input_storage_bytes(self) -> int:
        return self.prepared.estimated_storage_bytes

    @property
    def estimated_target_storage_bytes(self) -> int:
        return self.targets.estimated_storage_bytes

    def build_batch(self, sample_positions: torch.Tensor) -> PredictionBatch:
        input_batch = self.prepared.build_batch(sample_positions)
        return PredictionBatch(
            inputs=input_batch,
            targets=self.targets.build_batch(input_batch.sample_positions),
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PredictionPreparedRepresentation | None:
        prepared = self.prepared.to_device_storage(device)
        targets = self.targets.to_device_storage(device)
        if prepared is None or targets is None:
            return None
        return PredictionPreparedRepresentation(prepared=prepared, targets=targets)


@dataclass(frozen=True, slots=True)
class CompiledPredictionContract:
    prediction_id: str
    prediction_family_id: str
    training_metric_descriptors: tuple[MetricDescriptor, ...]
    progress_metric_descriptors: tuple[StageMetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    supported_workflows: frozenset[str]
    build_output_spec_fn: BuildOutputSpecFn
    prepare_targets_fn: PrepareTargetsFn
    compute_batch_loss_and_state_fn: ComputeBatchLossAndStateFn
    create_epoch_accumulator_fn: CreateEpochAccumulatorFn
    select_best_epoch_fn: SelectBestEpochFn
    allocate_decoded_offsets_fn: AllocateDecodedOffsetsFn
    decode_selected_offsets_into_fn: DecodeSelectedOffsetsIntoFn
    fit_training_state_fn: FitTrainingStateFn | None = None

    @property
    def semantics(self) -> PredictionSemantics:
        return PredictionSemantics(
            prediction_id=self.prediction_id,
            prediction_family_id=self.prediction_family_id,
            training_metric_descriptors=self.training_metric_descriptors,
            progress_metric_descriptors=self.progress_metric_descriptors,
            primary_metric_id=self.primary_metric_id,
            direction=self.direction,
            supported_workflows=self.supported_workflows,
        )

    @property
    def checkpoint_monitor(self) -> str:
        return f"validation_{self.primary_metric_id}"

    @property
    def early_stopping_monitor(self) -> str:
        return self.checkpoint_monitor

    def build_output_spec(self, max_candidate_slots: int) -> PredictionOutputSpec:
        return self.build_output_spec_fn(max_candidate_slots)

    def fit_training_state(
        self,
        store: CompiledProblemStore,
        train_sample_indices: IntVector,
    ) -> object | None:
        if self.fit_training_state_fn is None:
            return None
        return self.fit_training_state_fn(store, train_sample_indices)

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedPredictionTargets:
        return self.prepare_targets_fn(store, sample_indices)

    def compute_batch_loss_and_state(
        self,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
        *,
        training_state: object | None,
    ) -> tuple[torch.Tensor, object]:
        return self.compute_batch_loss_and_state_fn(outputs, targets, training_state)

    def create_epoch_accumulator(self, stage: str) -> EpochMetricAccumulator:
        return self.create_epoch_accumulator_fn(stage)

    def best_epoch(self, history: list[MetricSet]) -> int:
        return self.select_best_epoch_fn(history)

    def optimization_value(self, metrics: MetricSet) -> float:
        return metrics.require(self.primary_metric_id)

    def format_progress_metrics(self, metrics: MetricSet) -> tuple[StageMetricValue, ...]:
        return tuple(
            StageMetricValue(
                id=descriptor.id,
                value=format_compact_number(metrics.values[descriptor.id]),
            )
            for descriptor in self.progress_metric_descriptors
            if descriptor.id in metrics.values
        )

    def allocate_decoded_offsets(self, sample_count: int) -> object:
        return self.allocate_decoded_offsets_fn(sample_count)

    def decode_selected_offsets_into(
        self,
        predictions: object,
        sample_positions: torch.Tensor,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        self.decode_selected_offsets_into_fn(predictions, sample_positions, outputs, targets)


def bind_prediction_representation(
    prepared: PreparedRepresentation,
    *,
    targets: PreparedPredictionTargets,
) -> PredictionPreparedRepresentation:
    return PredictionPreparedRepresentation(prepared=prepared, targets=targets)

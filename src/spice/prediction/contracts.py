"""Compiled prediction contracts and batch wrappers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

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


class PredictionTargetBatch(Protocol):
    def to_device(self, device: torch.device) -> PredictionTargetBatch: ...


class PreparedPredictionTargets(Protocol):
    def build_batch(self, sample_positions: torch.Tensor) -> PredictionTargetBatch: ...


class EpochMetricAccumulator(Protocol):
    def update(self, batch_state: object) -> None: ...

    def snapshot(self) -> MetricSet: ...

    def finalize(self) -> MetricSet: ...


@dataclass(slots=True)
class PredictionBatch:
    inputs: ModelInputBatch
    targets: PredictionTargetBatch

    @property
    def sample_positions(self) -> torch.Tensor:
        return self.inputs.sample_positions

    def to_device(self, device: torch.device) -> PredictionBatch:
        return PredictionBatch(
            inputs=self.inputs.to_device(device),
            targets=self.targets.to_device(device),
        )

    def model_kwargs(self) -> Mapping[str, torch.Tensor]:
        return self.inputs.model_kwargs()


@dataclass(slots=True)
class PredictionPreparedRepresentation:
    prepared: PreparedRepresentation
    targets: PreparedPredictionTargets

    @property
    def representation_id(self) -> str:
        return self.prepared.representation_id

    @property
    def storage_mode_id(self) -> str:
        return self.prepared.storage_mode_id

    @property
    def batch_planner_id(self) -> str:
        return self.prepared.batch_planner_id

    def __len__(self) -> int:
        return len(self.prepared)

    def iter_batches(
        self,
        *,
        epoch: int,
        seed: int,
        shuffle: bool,
    ) -> Iterator[PredictionBatch]:
        for input_batch in self.prepared.iter_batches(
            epoch=epoch,
            seed=seed,
            shuffle=shuffle,
        ):
            yield PredictionBatch(
                inputs=input_batch,
                targets=self.targets.build_batch(input_batch.sample_positions),
            )


@dataclass(frozen=True, slots=True)
class CompiledPredictionContract:
    prediction_id: str
    prediction_family_id: str
    training_metric_descriptors: tuple[MetricDescriptor, ...]
    progress_metric_descriptors: tuple[StageMetricDescriptor, ...]
    primary_metric_id: str
    direction: Literal["maximize", "minimize"]
    supported_workflows: frozenset[str]

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
        raise NotImplementedError

    def fit_training_state(
        self,
        store: CompiledProblemStore,
        train_sample_indices: IntVector,
    ) -> object | None:
        del store, train_sample_indices
        return None

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedPredictionTargets:
        raise NotImplementedError

    def compute_batch_loss_and_state(
        self,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
        *,
        training_state: object | None,
    ) -> tuple[torch.Tensor, object]:
        raise NotImplementedError

    def create_epoch_accumulator(self, stage: str) -> EpochMetricAccumulator:
        raise NotImplementedError

    def best_epoch(self, history: list[MetricSet]) -> int:
        raise NotImplementedError

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
        raise NotImplementedError

    def decode_selected_offsets_into(
        self,
        predictions: object,
        sample_positions: torch.Tensor,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        raise NotImplementedError


def bind_prediction_representation(
    prepared: PreparedRepresentation,
    *,
    targets: PreparedPredictionTargets,
) -> PredictionPreparedRepresentation:
    return PredictionPreparedRepresentation(prepared=prepared, targets=targets)

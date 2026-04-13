"""Compiled prediction contracts and batch wrappers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.reporting import Reporter
from ..temporal.problem_store import CompiledProblemStore
from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionOutputSpec,
    PredictionSimulationSummary,
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
    objective_id: str
    metric_descriptors: tuple[MetricDescriptor, ...]
    primary_metric_id: str
    direction: str
    build_output_spec: Callable[[int], PredictionOutputSpec]
    prepare_targets: Callable[[CompiledProblemStore, IntVector], PreparedPredictionTargets]
    compute_batch_loss_and_state: Callable[
        [ModelOutputs, PredictionTargetBatch],
        tuple[torch.Tensor, object],
    ]
    summarize_epoch_metrics: Callable[[list[object]], MetricSet]
    best_epoch: Callable[[list[MetricSet]], int]
    objective_value: Callable[[MetricSet], float]
    allocate_prediction_buffer: Callable[[int], object]
    decode_into: Callable[[object, torch.Tensor, ModelOutputs, PredictionTargetBatch], None]
    replay: Callable[
        [CompiledProblemStore, object, IntVector, int, float, int, int, Reporter | None],
        PredictionSimulationSummary,
    ]
    supported_workflows: frozenset[str]

    @property
    def checkpoint_monitor(self) -> str:
        return f"validation_{self.primary_metric_id}"

    @property
    def early_stopping_monitor(self) -> str:
        return self.checkpoint_monitor


def bind_prediction_representation(
    prepared: PreparedRepresentation,
    *,
    targets: PreparedPredictionTargets,
) -> PredictionPreparedRepresentation:
    return PredictionPreparedRepresentation(prepared=prepared, targets=targets)

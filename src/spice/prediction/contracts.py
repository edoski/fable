"""Compiled prediction contracts and batch protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..semantics import PredictionSemantics
from ..temporal.problem_store import CompiledProblemStore
from ..temporal.realization import CompiledRealizationPolicyContract
from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionOutputSpec,
)

if TYPE_CHECKING:
    from ..modeling.models import ModelOutputs

IntVector = NDArray[np.int64]
BoolMatrix = NDArray[np.bool_]


def _coerce_cpu_int64_vector(
    values: torch.Tensor | IntVector,
    *,
    label: str,
) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return torch.from_numpy(values.astype(np.int64, copy=False))
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    return values.detach().to(device="cpu", dtype=torch.int64)


def _coerce_bool_matrix(
    values: torch.Tensor | BoolMatrix,
    *,
    label: str,
) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        if values.ndim != 2:
            raise ValueError(f"{label} must be two-dimensional")
        return torch.from_numpy(values.astype(np.bool_, copy=False))
    if values.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    return values.detach().to(dtype=torch.bool)


class DecodedOffsets:
    """Prediction-owned decoded offset buffer backed by a CPU int64 tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = _coerce_cpu_int64_vector(tensor, label="decoded_offsets")

    @classmethod
    def allocate(cls, sample_count: int) -> DecodedOffsets:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        return cls(torch.zeros(sample_count, dtype=torch.int64))

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def __len__(self) -> int:
        return int(self._tensor.shape[0])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecodedOffsets):
            return torch.equal(self._tensor, other._tensor)
        if isinstance(other, torch.Tensor):
            return torch.equal(
                self._tensor,
                _coerce_cpu_int64_vector(other, label="decoded_offsets"),
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            return self._tensor.tolist() == list(other)
        return NotImplemented

    def write(self, sample_positions: torch.Tensor, decoded: torch.Tensor) -> None:
        positions = _coerce_cpu_int64_vector(sample_positions, label="sample_positions")
        values = _coerce_cpu_int64_vector(decoded, label="decoded")
        if positions.shape != values.shape:
            raise ValueError("sample_positions and decoded must have matching shape")
        self._tensor.index_copy_(0, positions, values)

    def select(self, sample_positions: torch.Tensor | IntVector) -> IntVector:
        positions = _coerce_cpu_int64_vector(sample_positions, label="sample_positions")
        return self._tensor.index_select(0, positions).numpy()


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
    [CompiledProblemStore, IntVector, CompiledRealizationPolicyContract],
    object | None,
]
PrepareTargetsFn = Callable[
    [CompiledProblemStore, IntVector, CompiledRealizationPolicyContract],
    PreparedPredictionTargets,
]
ComputeBatchLossAndStateFn = Callable[
    [Any, PredictionTargetBatch, object | None],
    tuple[torch.Tensor, object],
]
CreateEpochAccumulatorFn = Callable[[], EpochMetricAccumulator]
DecodeSelectedOffsetsIntoFn = Callable[[DecodedOffsets, Any, "ActionSpaceDecodeContext"], None]


def masked_offset_argmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    mask = action_mask.detach().to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError("action_mask must match logits shape")
    if bool(torch.any(~mask.any(dim=1))):
        raise ValueError("action_mask must allow at least one action per sample")
    return logits.masked_fill(~mask, float("-inf")).argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class ActionSpaceDecodeContext:
    sample_positions: torch.Tensor
    action_mask: torch.Tensor

    def __post_init__(self) -> None:
        sample_positions = _coerce_cpu_int64_vector(
            self.sample_positions,
            label="sample_positions",
        )
        action_mask = _coerce_bool_matrix(
            self.action_mask,
            label="action_mask",
        )
        if sample_positions.shape[0] != action_mask.shape[0]:
            raise ValueError("sample_positions and action_mask must have matching rows")
        if action_mask.numel() > 0 and bool(torch.any(~action_mask.any(dim=1))):
            raise ValueError("action_mask must allow at least one action per sample")
        object.__setattr__(self, "sample_positions", sample_positions)
        object.__setattr__(self, "action_mask", action_mask)


def decode_context_from_batch(
    batch: ModelInputBatch,
) -> ActionSpaceDecodeContext:
    return ActionSpaceDecodeContext(
        sample_positions=batch.sample_positions,
        action_mask=batch.action_mask,
    )


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
    decode_selected_offsets_into_fn: DecodeSelectedOffsetsIntoFn
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
        realization_policy: CompiledRealizationPolicyContract,
    ) -> object | None:
        if self.fit_training_state_fn is None:
            return None
        return self.fit_training_state_fn(store, train_sample_indices, realization_policy)

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
        *,
        realization_policy: CompiledRealizationPolicyContract,
    ) -> PreparedPredictionTargets:
        return self.prepare_targets_fn(store, sample_indices, realization_policy)

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

    def allocate_decoded_offsets(self, sample_count: int) -> DecodedOffsets:
        return DecodedOffsets.allocate(sample_count)

    def decode_selected_offsets_into(
        self,
        predictions: DecodedOffsets,
        outputs: ModelOutputs,
        decode_context: ActionSpaceDecodeContext,
    ) -> None:
        self.decode_selected_offsets_into_fn(predictions, outputs, decode_context)

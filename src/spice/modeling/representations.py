"""Internal model-input representation execution helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from ..prediction import ModelInputBatch
from ..semantics import RepresentationSemantics
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore

IntVector = NDArray[np.int64]
_MAX_AUTOMATIC_MATERIALIZATION_BYTES = 8 * 1024**3
_CUDA_DEVICE_MATERIALIZATION_STAGING_BYTES = 256 * 1024**2
SEQUENCE_INPUT_REPRESENTATION_ID = "sequence_inputs"
BatchT = TypeVar("BatchT", bound=ModelInputBatch, covariant=True)


class SequenceInputBatch(NamedTuple):
    sample_positions: torch.Tensor
    inputs: torch.Tensor
    input_mask: torch.Tensor
    action_mask: torch.Tensor

    def to_device(self, device: torch.device) -> SequenceInputBatch:
        if (
            self.inputs.device == device
            and self.input_mask.device == device
            and self.action_mask.device == device
        ):
            return self
        non_blocking = device.type == "cuda"
        return SequenceInputBatch(
            sample_positions=self.sample_positions,
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            input_mask=self.input_mask.to(device, non_blocking=non_blocking),
            action_mask=self.action_mask.to(device, non_blocking=non_blocking),
        )

    def model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs,
            "input_mask": self.input_mask,
        }

    def pin_memory(self) -> SequenceInputBatch:
        if self.inputs.device.type != "cpu":
            return self
        return SequenceInputBatch(
            sample_positions=self.sample_positions.pin_memory(),
            inputs=self.inputs.pin_memory(),
            input_mask=self.input_mask.pin_memory(),
            action_mask=self.action_mask.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class RepresentationRuntimeContext:
    batch_size: int
    available_host_memory_bytes: int
    available_device_memory_bytes: int | None = None

    def with_device_memory_budget(
        self,
        available_device_memory_bytes: int | None,
    ) -> RepresentationRuntimeContext:
        if available_device_memory_bytes is not None and available_device_memory_bytes < 0:
            raise ValueError("available_device_memory_bytes must be non-negative")
        return RepresentationRuntimeContext(
            batch_size=self.batch_size,
            available_host_memory_bytes=self.available_host_memory_bytes,
            available_device_memory_bytes=available_device_memory_bytes,
        )


class PreparedRepresentation(Protocol[BatchT]):
    @property
    def sample_count(self) -> int: ...

    @property
    def batch_signatures(self) -> IntVector: ...

    @property
    def estimated_storage_bytes(self) -> int: ...

    def build_batch(self, sample_positions: torch.Tensor) -> BatchT: ...

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedRepresentation[BatchT]: ...


@dataclass(frozen=True, slots=True)
class CompiledRepresentationContract:
    """Compiled model-input representation seam used by training and inference only."""

    prepare_impl: Callable[..., PreparedRepresentation[ModelInputBatch]]

    @property
    def semantics(self) -> RepresentationSemantics:
        return RepresentationSemantics(representation_id=SEQUENCE_INPUT_REPRESENTATION_ID)

    @property
    def representation_id(self) -> str:
        return SEQUENCE_INPUT_REPRESENTATION_ID

    def prepare(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
        *,
        execution_policy: CompiledExecutionPolicyContract,
        action_space: PreparedActionSpace,
        runtime_context: RepresentationRuntimeContext,
    ) -> PreparedRepresentation[ModelInputBatch]:
        return self.prepare_impl(
            store,
            sample_indices,
            execution_policy=execution_policy,
            action_space=action_space,
            runtime_context=runtime_context,
        )


def sequence_input_contract() -> CompiledRepresentationContract:
    return _SEQUENCE_INPUT_CONTRACT


def validate_representation_id(representation_id: str) -> None:
    if representation_id != SEQUENCE_INPUT_REPRESENTATION_ID:
        raise ValueError("representation_id must be sequence_inputs")


def build_sequence_input_batch(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    action_mask: NDArray[np.bool_],
    sample_positions: IntVector | torch.Tensor | None = None,
    max_context_length: int | None = None,
) -> SequenceInputBatch:
    if sample_indices.size == 0:
        raise ValueError("Sequence batches require at least one sample")
    sample_indices = sample_indices.astype(np.int64, copy=False)
    context = store.context_windows(sample_indices)
    input_lengths = context.context_lengths
    batch_size = int(sample_indices.shape[0])
    if sample_positions is None:
        resolved_positions = np.arange(batch_size, dtype=np.int64)
    elif isinstance(sample_positions, torch.Tensor):
        resolved_positions = sample_positions.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        resolved_positions = sample_positions.astype(np.int64, copy=False)
    if resolved_positions.shape[0] != batch_size:
        raise ValueError("sample_positions must match sample_indices length")
    resolved_max_context = (
        int(input_lengths.max()) if max_context_length is None else int(max_context_length)
    )
    if np.any(input_lengths > resolved_max_context):
        raise ValueError("max_context_length is too small for the requested batch")

    inputs = np.zeros((batch_size, resolved_max_context, store.n_features), dtype=np.float32)
    input_mask = np.zeros((batch_size, resolved_max_context), dtype=np.bool_)
    resolved_action_mask = action_mask.astype(np.bool_, copy=False)
    if resolved_action_mask.shape != (batch_size, store.max_candidate_slots):
        raise ValueError("action_mask must match sample count and action width")
    for row, (anchor_row, context_start) in enumerate(
        zip(context.anchor_rows, context.context_start_rows, strict=True)
    ):
        anchor_row = int(anchor_row)
        context_start = int(context_start)
        sequence = store.feature_matrix[context_start : anchor_row + 1]
        inputs[row, : sequence.shape[0], :] = sequence
        input_mask[row, : sequence.shape[0]] = True

    return SequenceInputBatch(
        sample_positions=torch.from_numpy(np.ascontiguousarray(resolved_positions)),
        inputs=torch.from_numpy(inputs),
        input_mask=torch.from_numpy(input_mask),
        action_mask=torch.from_numpy(np.ascontiguousarray(resolved_action_mask)),
    )


@dataclass(frozen=True, slots=True)
class _SequenceInputLayout:
    sample_indices: IntVector
    context_lengths: IntVector
    max_context_length: int


@dataclass(slots=True)
class _StreamingSequenceInputRepresentation:
    store: CompiledProblemStore
    action_space: PreparedActionSpace
    layout: _SequenceInputLayout
    batch_size: int

    @property
    def sample_count(self) -> int:
        return int(self.layout.sample_indices.shape[0])

    @property
    def batch_signatures(self) -> IntVector:
        return self.layout.context_lengths

    @property
    def estimated_storage_bytes(self) -> int:
        return _dense_sequence_input_storage_bytes(
            self.layout,
            n_features=self.store.n_features,
            max_candidate_slots=self.store.max_candidate_slots,
        )

    @property
    def host_storage_mode(self) -> Literal["host_streaming"]:
        return "host_streaming"

    def build_batch(self, sample_positions: torch.Tensor) -> ModelInputBatch:
        positions = sample_positions.detach().cpu().numpy().astype(np.int64, copy=False)
        batch_sample_indices = self.layout.sample_indices[positions]
        return build_sequence_input_batch(
            self.store,
            batch_sample_indices,
            action_mask=self.action_space.action_mask[positions],
            sample_positions=sample_positions,
            max_context_length=self.layout.max_context_length,
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedRepresentation[ModelInputBatch]:
        _require_cuda_storage_device(device)
        return _materialize_sequence_input_to_device(
            self.store,
            self.action_space,
            self.layout,
            batch_size=self.batch_size,
            device=device,
        )


@dataclass(slots=True)
class _MaterializedSequenceInputRepresentation:
    inputs: torch.Tensor
    input_mask: torch.Tensor
    action_mask: torch.Tensor
    layout: _SequenceInputLayout
    batch_size: int

    @property
    def sample_count(self) -> int:
        return int(self.layout.sample_indices.shape[0])

    @property
    def batch_signatures(self) -> IntVector:
        return self.layout.context_lengths

    @property
    def estimated_storage_bytes(self) -> int:
        return (
            self.inputs.element_size() * self.inputs.numel()
            + self.input_mask.element_size() * self.input_mask.numel()
            + self.action_mask.element_size() * self.action_mask.numel()
        )

    @property
    def host_storage_mode(self) -> Literal["host_materialized"]:
        return "host_materialized"

    def build_batch(self, sample_positions: torch.Tensor) -> ModelInputBatch:
        positions = sample_positions.detach().cpu().to(dtype=torch.int64, copy=False)
        index = positions.to(device=self.inputs.device)
        return SequenceInputBatch(
            sample_positions=positions,
            inputs=self.inputs.index_select(0, index),
            input_mask=self.input_mask.index_select(0, index),
            action_mask=self.action_mask.index_select(0, index),
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedRepresentation[ModelInputBatch]:
        if (
            self.inputs.device == device
            and self.input_mask.device == device
            and self.action_mask.device == device
        ):
            return self
        _require_cuda_storage_device(device)
        return _MaterializedSequenceInputRepresentation(
            inputs=self.inputs.to(device, non_blocking=True),
            input_mask=self.input_mask.to(device, non_blocking=True),
            action_mask=self.action_mask.to(device, non_blocking=True),
            layout=self.layout,
            batch_size=self.batch_size,
        )


def _prepare_sequence_input(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    execution_policy: CompiledExecutionPolicyContract,
    action_space: PreparedActionSpace,
    runtime_context: RepresentationRuntimeContext,
) -> PreparedRepresentation[ModelInputBatch]:
    del execution_policy
    if runtime_context.batch_size <= 0:
        raise ValueError("runtime_context.batch_size must be positive")
    layout = _sequence_input_layout(store, sample_indices)
    dense_storage_bytes = _dense_sequence_input_storage_bytes(
        layout,
        n_features=store.n_features,
        max_candidate_slots=store.max_candidate_slots,
    )
    materialization_budget = min(
        _MAX_AUTOMATIC_MATERIALIZATION_BYTES,
        max(0, runtime_context.available_host_memory_bytes // 5),
    )
    if dense_storage_bytes <= materialization_budget:
        return _materialize_sequence_input(
            store,
            action_space,
            layout,
            batch_size=runtime_context.batch_size,
        )
    return _StreamingSequenceInputRepresentation(
        store=store,
        action_space=action_space,
        layout=layout,
        batch_size=runtime_context.batch_size,
    )


_SEQUENCE_INPUT_CONTRACT = CompiledRepresentationContract(
    prepare_impl=_prepare_sequence_input,
)


def _sequence_input_layout(
    store: CompiledProblemStore,
    sample_indices: IntVector,
) -> _SequenceInputLayout:
    if sample_indices.size == 0:
        raise ValueError("Prepared representations require at least one sample")
    resolved_sample_indices = sample_indices.astype(np.int64, copy=False)
    context = store.context_windows(resolved_sample_indices)
    return _SequenceInputLayout(
        sample_indices=resolved_sample_indices,
        context_lengths=context.context_lengths,
        max_context_length=int(context.context_lengths.max()),
    )


def _dense_sequence_input_storage_bytes(
    layout: _SequenceInputLayout,
    *,
    n_features: int,
    max_candidate_slots: int,
) -> int:
    sample_count = int(layout.sample_indices.shape[0])
    inputs_bytes = (
        sample_count * layout.max_context_length * n_features * np.dtype(np.float32).itemsize
    )
    input_mask_bytes = sample_count * layout.max_context_length * np.dtype(np.bool_).itemsize
    action_mask_bytes = sample_count * max_candidate_slots * np.dtype(np.bool_).itemsize
    return inputs_bytes + input_mask_bytes + action_mask_bytes


def _materialize_sequence_input(
    store: CompiledProblemStore,
    action_space: PreparedActionSpace,
    layout: _SequenceInputLayout,
    *,
    batch_size: int,
) -> _MaterializedSequenceInputRepresentation:
    sample_count = int(layout.sample_indices.shape[0])
    inputs = np.zeros(
        (sample_count, layout.max_context_length, store.n_features),
        dtype=np.float32,
    )
    input_mask = np.zeros((sample_count, layout.max_context_length), dtype=np.bool_)
    action_mask = np.zeros((sample_count, store.max_candidate_slots), dtype=np.bool_)
    _fill_dense_sequence_input_rows(
        store,
        action_space,
        layout,
        row_start=0,
        row_stop=sample_count,
        inputs=inputs,
        input_mask=input_mask,
        action_mask=action_mask,
    )

    return _MaterializedSequenceInputRepresentation(
        inputs=torch.from_numpy(inputs),
        input_mask=torch.from_numpy(input_mask),
        action_mask=torch.from_numpy(action_mask),
        layout=layout,
        batch_size=batch_size,
    )


def _materialize_sequence_input_to_device(
    store: CompiledProblemStore,
    action_space: PreparedActionSpace,
    layout: _SequenceInputLayout,
    *,
    batch_size: int,
    device: torch.device,
) -> _MaterializedSequenceInputRepresentation:
    _require_cuda_storage_device(device)
    sample_count = int(layout.sample_indices.shape[0])
    inputs = torch.zeros(
        (sample_count, layout.max_context_length, store.n_features),
        dtype=torch.float32,
        device=device,
    )
    input_mask = torch.zeros(
        (sample_count, layout.max_context_length),
        dtype=torch.bool,
        device=device,
    )
    action_mask = torch.zeros(
        (sample_count, store.max_candidate_slots),
        dtype=torch.bool,
        device=device,
    )
    row_bytes = (
        layout.max_context_length * store.n_features * np.dtype(np.float32).itemsize
        + layout.max_context_length * np.dtype(np.bool_).itemsize
        + store.max_candidate_slots * np.dtype(np.bool_).itemsize
    )
    chunk_rows = max(
        1,
        min(
            sample_count,
            _CUDA_DEVICE_MATERIALIZATION_STAGING_BYTES // max(1, row_bytes),
        ),
    )
    for row_start in range(0, sample_count, chunk_rows):
        row_stop = min(sample_count, row_start + chunk_rows)
        chunk_inputs = np.zeros(
            (row_stop - row_start, layout.max_context_length, store.n_features),
            dtype=np.float32,
        )
        chunk_input_mask = np.zeros(
            (row_stop - row_start, layout.max_context_length),
            dtype=np.bool_,
        )
        chunk_action_mask = np.zeros(
            (row_stop - row_start, store.max_candidate_slots),
            dtype=np.bool_,
        )
        _fill_dense_sequence_input_rows(
            store,
            action_space,
            layout,
            row_start=row_start,
            row_stop=row_stop,
            inputs=chunk_inputs,
            input_mask=chunk_input_mask,
            action_mask=chunk_action_mask,
        )
        chunk_inputs_tensor = torch.from_numpy(chunk_inputs).pin_memory()
        chunk_mask_tensor = torch.from_numpy(chunk_input_mask).pin_memory()
        chunk_action_mask_tensor = torch.from_numpy(chunk_action_mask).pin_memory()
        inputs[row_start:row_stop].copy_(chunk_inputs_tensor, non_blocking=True)
        input_mask[row_start:row_stop].copy_(chunk_mask_tensor, non_blocking=True)
        action_mask[row_start:row_stop].copy_(chunk_action_mask_tensor, non_blocking=True)
    return _MaterializedSequenceInputRepresentation(
        inputs=inputs,
        input_mask=input_mask,
        action_mask=action_mask,
        layout=layout,
        batch_size=batch_size,
    )


def _require_cuda_storage_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("prepared representation storage migration requires CUDA")


def _fill_dense_sequence_input_rows(
    store: CompiledProblemStore,
    action_space: PreparedActionSpace,
    layout: _SequenceInputLayout,
    *,
    row_start: int,
    row_stop: int,
    inputs: np.ndarray,
    input_mask: np.ndarray,
    action_mask: np.ndarray,
) -> None:
    sample_indices = layout.sample_indices[row_start:row_stop]
    context = store.context_windows(sample_indices)
    action_mask[:, :] = action_space.action_mask[row_start:row_stop]
    for output_row, (anchor_row, context_start) in enumerate(
        zip(context.anchor_rows, context.context_start_rows, strict=True)
    ):
        anchor_row = int(anchor_row)
        context_start = int(context_start)
        sequence = store.feature_matrix[context_start : anchor_row + 1]
        inputs[output_row, : sequence.shape[0], :] = sequence
        input_mask[output_row, : sequence.shape[0]] = True

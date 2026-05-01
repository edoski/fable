"""Batch planning for training and inference."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Sampler

from ..prediction import CompiledPredictionContract
from ..prediction.contracts import ModelInputBatch, PredictionBatch, PreparedPredictionTargets
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from .representations import (
    CompiledRepresentationContract,
    PreparedRepresentation,
    RepresentationRuntimeContext,
)

IntVector = NDArray[np.int64]
BatchT = TypeVar("BatchT", covariant=True)
StorageMode = Literal["host_streaming", "host_materialized", "cuda_materialized"]
HostStorageMode = Literal["host_streaming", "host_materialized"]


class BatchSource(Protocol[BatchT]):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[BatchT]: ...


@dataclass(frozen=True, slots=True)
class BatchPlan(Generic[BatchT]):
    source: BatchSource[BatchT]
    storage_mode: StorageMode
    sample_count: int
    batch_count: int
    estimated_storage_bytes: int


class PreparedBatchRepresentation(Protocol[BatchT]):
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
    ) -> PreparedBatchRepresentation[BatchT]: ...


class _SamplePositionDataset(Dataset[int]):
    def __init__(self, sample_count: int) -> None:
        self._sample_count = sample_count

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: int) -> int:
        return int(index)


class _PositionBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        batch_signatures: IntVector,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self._batch_signatures = batch_signatures.astype(np.int64, copy=False)
        self._batch_size = batch_size
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0

    def __len__(self) -> int:
        return math.ceil(int(self._batch_signatures.shape[0]) / self._batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        order = _ordered_sample_positions(
            self._batch_signatures,
            batch_size=self._batch_size,
            epoch=self._epoch if self._shuffle else 0,
            seed=self._seed,
            shuffle=self._shuffle,
        )
        if self._shuffle:
            self._epoch += 1
        for offset in range(0, int(order.shape[0]), self._batch_size):
            yield order[offset : offset + self._batch_size].tolist()


def _ordered_sample_positions(
    batch_signatures: IntVector,
    *,
    batch_size: int,
    epoch: int,
    seed: int,
    shuffle: bool,
) -> IntVector:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    signatures = batch_signatures.astype(np.int64, copy=False)
    order = np.argsort(signatures, kind="stable").astype(np.int64, copy=False)
    if not shuffle or order.size == 0:
        return order

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    batches = [
        order[offset : offset + batch_size].copy()
        for offset in range(0, int(order.shape[0]), batch_size)
    ]
    shuffled_batches = [rng.permutation(batches[index]) for index in rng.permutation(len(batches))]
    return np.concatenate(shuffled_batches).astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class _HostBatchCollator(Generic[BatchT]):
    prepared: PreparedBatchRepresentation[BatchT]

    def __call__(self, sample_positions: Sequence[int]) -> BatchT:
        index = torch.as_tensor(sample_positions, dtype=torch.int64)
        return self.prepared.build_batch(index)


class _HostDataLoaderBatchSource(Generic[BatchT]):
    def __init__(self, loader: DataLoader[BatchT], batch_sampler: _PositionBatchSampler) -> None:
        self._loader = loader
        self._batch_sampler = batch_sampler

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def __iter__(self) -> Iterator[BatchT]:
        return iter(self._loader)


@dataclass(frozen=True, slots=True)
class _HostLoaderWorkerSettings:
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int | None


class _DeviceResidentBatchSource(Generic[BatchT]):
    def __init__(
        self,
        *,
        prepared: PreparedBatchRepresentation[BatchT],
        batch_sampler: _PositionBatchSampler,
    ) -> None:
        self.prepared = prepared
        self.batch_sampler = batch_sampler

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[BatchT]:
        for sample_positions in self.batch_sampler:
            yield self.prepared.build_batch(torch.as_tensor(sample_positions, dtype=torch.int64))


@dataclass(slots=True)
class _PreparedPredictionBatches:
    prepared: PreparedRepresentation
    targets: PreparedPredictionTargets

    @property
    def sample_count(self) -> int:
        return self.prepared.sample_count

    @property
    def batch_signatures(self) -> IntVector:
        return self.prepared.batch_signatures

    @property
    def estimated_storage_bytes(self) -> int:
        return self.prepared.estimated_storage_bytes + self.targets.estimated_storage_bytes

    @property
    def host_storage_mode(self) -> HostStorageMode:
        return _host_storage_mode(self.prepared)

    def build_batch(self, sample_positions: torch.Tensor) -> PredictionBatch:
        input_batch = self.prepared.build_batch(sample_positions)
        return PredictionBatch(
            inputs=input_batch,
            targets=self.targets.build_batch(input_batch.sample_positions),
        )

    def to_device_storage(self, device: torch.device) -> _PreparedPredictionBatches:
        return _PreparedPredictionBatches(
            prepared=self.prepared.to_device_storage(device),
            targets=self.targets.to_device_storage(device),
        )


def _build_plan(
    prepared: PreparedBatchRepresentation[BatchT],
    *,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool,
) -> BatchPlan[BatchT]:
    source, storage_mode = _build_batch_source(
        prepared,
        required_bytes=prepared.estimated_storage_bytes,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=shuffle,
    )
    return BatchPlan(
        source=cast(BatchSource[BatchT], source),
        storage_mode=storage_mode,
        sample_count=prepared.sample_count,
        batch_count=len(source),
        estimated_storage_bytes=prepared.estimated_storage_bytes,
    )


def _prepare_model_representation(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    execution_policy: CompiledExecutionPolicyContract,
    action_space: PreparedActionSpace,
    runtime_context: RepresentationRuntimeContext,
) -> PreparedRepresentation:
    return representation_contract.prepare(
        store,
        sample_indices,
        execution_policy=execution_policy,
        action_space=action_space,
        runtime_context=runtime_context,
    )


def _prepare_action_space(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    execution_policy: CompiledExecutionPolicyContract,
) -> PreparedActionSpace:
    action_space = execution_policy.prepare_action_space(store, sample_indices)
    expected_shape = (int(sample_indices.shape[0]), store.max_candidate_slots)
    if action_space.action_mask.shape != expected_shape:
        raise ValueError(
            "prepared Action Space action_mask shape does not match sample count "
            "and action width"
        )
    return action_space


def build_prediction_batch_plan(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool = False,
) -> BatchPlan[PredictionBatch]:
    action_space = _prepare_action_space(
        store,
        sample_indices,
        execution_policy=execution_policy,
    )
    prepared = _prepare_model_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        execution_policy=execution_policy,
        action_space=action_space,
        runtime_context=runtime_context,
    )
    targets = prediction_contract.prepare_targets(
        store,
        sample_indices,
        execution_policy=execution_policy,
        action_space=action_space,
    )
    return _build_plan(
        _PreparedPredictionBatches(prepared=prepared, targets=targets),
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=shuffle,
    )


def build_model_input_batch_plan(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    *,
    representation_contract: CompiledRepresentationContract,
    execution_policy: CompiledExecutionPolicyContract,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
) -> BatchPlan[ModelInputBatch]:
    action_space = _prepare_action_space(
        store,
        sample_indices,
        execution_policy=execution_policy,
    )
    prepared = _prepare_model_representation(
        store,
        sample_indices,
        representation_contract=representation_contract,
        execution_policy=execution_policy,
        action_space=action_space,
        runtime_context=runtime_context,
    )
    return _build_plan(
        prepared,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=False,
    )


def _build_batch_source(
    prepared: PreparedBatchRepresentation[BatchT],
    *,
    required_bytes: int,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool,
) -> tuple[
    _HostDataLoaderBatchSource[BatchT] | _DeviceResidentBatchSource[BatchT],
    StorageMode,
]:
    batch_sampler = _PositionBatchSampler(
        batch_signatures=prepared.batch_signatures,
        batch_size=runtime_context.batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    if _should_use_device_resident(
        required_bytes=required_bytes,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
    ):
        try:
            return (
                _DeviceResidentBatchSource(
                    prepared=prepared.to_device_storage(resolved_device),
                    batch_sampler=batch_sampler,
                ),
                "cuda_materialized",
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return (
        _build_host_dataloader_source(
            prepared,
            batch_sampler=batch_sampler,
            resolved_device=resolved_device,
        ),
        _host_storage_mode(prepared),
    )


def _host_storage_mode(prepared: object) -> HostStorageMode:
    return getattr(prepared, "host_storage_mode", "host_materialized")


def _build_host_dataloader_source(
    prepared: PreparedBatchRepresentation[BatchT],
    *,
    batch_sampler: _PositionBatchSampler,
    resolved_device: torch.device,
) -> _HostDataLoaderBatchSource[BatchT]:
    worker_settings = _resolve_host_loader_worker_settings()
    loader = DataLoader(
        _SamplePositionDataset(prepared.sample_count),
        batch_sampler=batch_sampler,
        collate_fn=_HostBatchCollator(prepared),
        num_workers=worker_settings.num_workers,
        persistent_workers=worker_settings.persistent_workers,
        prefetch_factor=worker_settings.prefetch_factor,
        pin_memory=_should_pin_host_memory(resolved_device),
    )
    return _HostDataLoaderBatchSource(
        loader=cast(DataLoader[BatchT], loader),
        batch_sampler=batch_sampler,
    )


def _resolve_host_loader_worker_settings() -> _HostLoaderWorkerSettings:
    requested_workers = os.environ.get("SPICE_DATALOADER_WORKERS")
    if requested_workers is not None:
        return _build_host_loader_worker_settings(
            num_workers=_parse_worker_override(requested_workers)
        )

    cpu_budget = _slurm_cpu_budget()
    if cpu_budget is None or cpu_budget <= 2:
        return _build_host_loader_worker_settings(num_workers=0)
    return _build_host_loader_worker_settings(num_workers=min(8, cpu_budget - 2))


def _parse_worker_override(raw_value: str) -> int:
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError("SPICE_DATALOADER_WORKERS must be an integer") from exc
    if parsed < 0:
        raise ValueError("SPICE_DATALOADER_WORKERS must be non-negative")
    return parsed


def _slurm_cpu_budget() -> int | None:
    raw_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _build_host_loader_worker_settings(*, num_workers: int) -> _HostLoaderWorkerSettings:
    if num_workers <= 0:
        return _HostLoaderWorkerSettings(
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
        )
    return _HostLoaderWorkerSettings(
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4 if num_workers >= 4 else 2,
    )


def _should_pin_host_memory(resolved_device: torch.device) -> bool:
    return resolved_device.type == "cuda" and torch.cuda.is_available()


def _should_use_device_resident(
    *,
    required_bytes: int,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
) -> bool:
    if resolved_device.type != "cuda":
        return False
    available_device_memory_bytes = runtime_context.available_device_memory_bytes
    if available_device_memory_bytes is None or available_device_memory_bytes <= 0:
        return False
    return required_bytes <= available_device_memory_bytes

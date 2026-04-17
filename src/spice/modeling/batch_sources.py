"""Internal batch-source planning for training and evaluation."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Sampler

from ..prediction.contracts import PredictionBatch, PredictionPreparedRepresentation
from .representations import RepresentationRuntimeContext

IntVector = NDArray[np.int64]
_CUDA_DEVICE_RESIDENT_BUDGET_FRACTION = 0.5
_CUDA_HOST_DATALOADER_MAX_WORKERS = 4
_CUDA_HOST_DATALOADER_PREFETCH_FACTOR = 2


class PreparedBatchSource(Protocol):
    @property
    def loader_strategy_id(self) -> str: ...

    @property
    def input_storage_mode_id(self) -> str: ...

    @property
    def target_storage_mode_id(self) -> str: ...

    @property
    def batch_planner_id(self) -> str: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[PredictionBatch]: ...


@dataclass(frozen=True, slots=True)
class BatchSourcePlan:
    source: PreparedBatchSource
    loader_strategy_id: str
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str


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
    epoch: int,
    seed: int,
    shuffle: bool,
) -> IntVector:
    order = np.arange(batch_signatures.shape[0], dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        order = rng.permutation(order)
    signatures = batch_signatures[order].astype(np.int64, copy=False)
    return order[np.argsort(signatures, kind="stable")]


@dataclass(frozen=True, slots=True)
class _HostPredictionBatchCollator:
    prepared: PredictionPreparedRepresentation

    def __call__(self, sample_positions: Sequence[int]) -> PredictionBatch:
        index = torch.as_tensor(sample_positions, dtype=torch.int64)
        return self.prepared.build_batch(index)


@dataclass(slots=True)
class HostDataLoaderBatchSource:
    _loader: DataLoader[PredictionBatch]
    _batch_sampler: _PositionBatchSampler
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str
    loader_strategy_id: str = "host_dataloader"

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def __iter__(self) -> Iterator[PredictionBatch]:
        return iter(self._loader)


@dataclass(slots=True)
class DeviceResidentBatchSource:
    prepared: PredictionPreparedRepresentation
    batch_sampler: _PositionBatchSampler
    input_storage_mode_id: str
    target_storage_mode_id: str
    batch_planner_id: str
    loader_strategy_id: str = "device_resident"

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[PredictionBatch]:
        for sample_positions in self.batch_sampler:
            yield self.prepared.build_batch(
                torch.as_tensor(sample_positions, dtype=torch.int64)
            )


def plan_batch_source(
    prepared: PredictionPreparedRepresentation,
    *,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool,
) -> BatchSourcePlan:
    if _should_use_device_resident(
        prepared,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
    ):
        device_prepared = prepared.to_device_storage(resolved_device)
        if device_prepared is not None:
            source: PreparedBatchSource = DeviceResidentBatchSource(
                prepared=device_prepared,
                batch_sampler=_PositionBatchSampler(
                    batch_signatures=device_prepared.batch_signatures,
                    batch_size=runtime_context.batch_size,
                    seed=seed,
                    shuffle=shuffle,
                ),
                input_storage_mode_id=device_prepared.input_storage_mode_id,
                target_storage_mode_id=device_prepared.target_storage_mode_id,
                batch_planner_id=device_prepared.batch_planner_id,
            )
            return BatchSourcePlan(
                source=source,
                loader_strategy_id=source.loader_strategy_id,
                input_storage_mode_id=source.input_storage_mode_id,
                target_storage_mode_id=source.target_storage_mode_id,
                batch_planner_id=source.batch_planner_id,
            )

    source = _build_host_dataloader_source(
        prepared,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=seed,
        shuffle=shuffle,
    )
    return BatchSourcePlan(
        source=source,
        loader_strategy_id=source.loader_strategy_id,
        input_storage_mode_id=source.input_storage_mode_id,
        target_storage_mode_id=source.target_storage_mode_id,
        batch_planner_id=source.batch_planner_id,
    )


def _build_host_dataloader_source(
    prepared: PredictionPreparedRepresentation,
    *,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    seed: int,
    shuffle: bool,
) -> HostDataLoaderBatchSource:
    batch_sampler = _PositionBatchSampler(
        batch_signatures=prepared.batch_signatures,
        batch_size=runtime_context.batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    num_workers = _resolve_host_dataloader_workers(resolved_device)
    loader_kwargs: dict[str, object] = {
        "batch_sampler": batch_sampler,
        "collate_fn": _HostPredictionBatchCollator(prepared),
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = _CUDA_HOST_DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(
        _SamplePositionDataset(prepared.sample_count),
        **loader_kwargs,
    )
    return HostDataLoaderBatchSource(
        _loader=loader,
        _batch_sampler=batch_sampler,
        input_storage_mode_id=prepared.input_storage_mode_id,
        target_storage_mode_id=prepared.target_storage_mode_id,
        batch_planner_id=prepared.batch_planner_id,
    )


def _resolve_host_dataloader_workers(resolved_device: torch.device) -> int:
    if resolved_device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(1, min(_CUDA_HOST_DATALOADER_MAX_WORKERS, cpu_count // 2))


def _should_use_device_resident(
    prepared: PredictionPreparedRepresentation,
    *,
    runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
) -> bool:
    if resolved_device.type != "cuda":
        return False
    available_device_memory_bytes = runtime_context.available_device_memory_bytes
    if available_device_memory_bytes is None or available_device_memory_bytes <= 0:
        return False
    if prepared.input_storage_mode_id.startswith("streaming"):
        return False
    required_bytes = (
        prepared.estimated_input_storage_bytes + prepared.estimated_target_storage_bytes
    )
    return required_bytes <= available_device_memory_bytes


def resolve_available_device_memory_budget(resolved_device: torch.device) -> int | None:
    if resolved_device.type != "cuda":
        return None
    device_index = (
        torch.cuda.current_device() if resolved_device.index is None else resolved_device.index
    )
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    return int(free_bytes * _CUDA_DEVICE_RESIDENT_BUDGET_FRACTION)

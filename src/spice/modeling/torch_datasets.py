"""PyTorch dataset adapters."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from ..data.datasets import TemporalDatasetStore

IntVector = NDArray[np.int64]


class SequenceBatch(NamedTuple):
    inputs: torch.Tensor
    class_label: torch.Tensor
    target_log_fee: torch.Tensor
    action_log_fees: torch.Tensor
    next_block_log_fee: torch.Tensor
    optimal_log_fee: torch.Tensor


def build_sequence_batch(
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    *,
    sequence_view: NDArray[np.float32],
    lookback_steps: int,
) -> SequenceBatch:
    if sample_indices.size == 0:
        raise ValueError("Sequence batches require at least one sample")
    sample_indices = sample_indices.astype(np.int64, copy=False)
    sequence_starts = store.sample_row_indices[sample_indices] - lookback_steps + 1
    return SequenceBatch(
        inputs=torch.from_numpy(np.ascontiguousarray(sequence_view[sequence_starts])),
        class_label=torch.from_numpy(
            np.ascontiguousarray(store.class_labels[sample_indices].astype(np.int64, copy=False))
        ),
        target_log_fee=torch.from_numpy(
            np.ascontiguousarray(
                store.target_log_fee[sample_indices].astype(np.float32, copy=False)
            )
        ),
        action_log_fees=torch.from_numpy(
            np.ascontiguousarray(
                store.action_log_fees[sample_indices].astype(np.float32, copy=False)
            )
        ),
        next_block_log_fee=torch.from_numpy(
            np.ascontiguousarray(
                store.next_block_log_fee[sample_indices].astype(np.float32, copy=False)
            )
        ),
        optimal_log_fee=torch.from_numpy(
            np.ascontiguousarray(
                store.optimal_log_fee[sample_indices].astype(np.float32, copy=False)
            )
        ),
    )


class SequenceBatchLoader:
    """Batch-native sequence loader over an array-backed temporal dataset store."""

    def __init__(
        self,
        store: TemporalDatasetStore,
        sample_indices: IntVector,
        *,
        lookback_steps: int,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        if sample_indices.size == 0:
            raise ValueError("SequenceBatchLoader requires at least one sample")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.store = store
        self.sample_indices = sample_indices.astype(np.int64, copy=False)
        self.lookback_steps = lookback_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        window_view = np.lib.stride_tricks.sliding_window_view(
            store.feature_matrix,
            window_shape=(lookback_steps, store.n_features),
        )
        self._sequence_view = window_view[:, 0].astype(np.float32, copy=False)

    def __len__(self) -> int:
        return math.ceil(int(self.sample_indices.shape[0]) / self.batch_size)

    def __iter__(self) -> Iterator[SequenceBatch]:
        order = self.sample_indices
        if self.shuffle:
            order = np.random.permutation(order)
        for offset in range(0, int(order.shape[0]), self.batch_size):
            batch_indices = order[offset : offset + self.batch_size]
            yield build_sequence_batch(
                self.store,
                batch_indices,
                sequence_view=self._sequence_view,
                lookback_steps=self.lookback_steps,
            )


def move_batch_to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(*(tensor.to(device) for tensor in batch))


def build_class_weights(
    class_labels: IntVector,
    sample_indices: IntVector,
    action_count: int,
) -> torch.Tensor:
    if sample_indices.size == 0:
        raise ValueError("Cannot build class weights for an empty sample selection")
    selected_labels = class_labels[sample_indices]
    counts = np.bincount(selected_labels, minlength=action_count)
    if counts.shape[0] != action_count:
        raise ValueError("class label space does not match action_count")
    if np.any(counts == 0):
        missing = [str(index) for index, count in enumerate(counts) if count == 0]
        raise ValueError(
            "Training split is missing at least one action class: " + ", ".join(missing)
        )
    return torch.from_numpy((1.0 / counts.astype(np.float32)).copy())

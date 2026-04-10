"""PyTorch dataset adapters."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..data.datasets import TemporalDatasetStore

IntVector = NDArray[np.int64]
SequenceBatch = dict[str, torch.Tensor]


class SequenceDataset(Dataset[SequenceBatch]):
    """Lazy tensor adapter over an array-backed temporal dataset store."""

    def __init__(
        self,
        store: TemporalDatasetStore,
        sample_indices: IntVector,
        *,
        lookback_steps: int,
    ) -> None:
        if sample_indices.size == 0:
            raise ValueError("SequenceDataset requires at least one sample")
        self.store = store
        self.sample_indices = sample_indices
        self.lookback_steps = lookback_steps

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> SequenceBatch:
        sample_index = int(self.sample_indices[index])
        anchor_row_index = int(self.store.anchor_row_indices[sample_index])
        sequence_start = anchor_row_index - self.lookback_steps + 1
        return {
            "inputs": torch.from_numpy(
                self.store.feature_matrix[sequence_start : anchor_row_index + 1]
            ),
            "class_label": torch.tensor(self.store.class_labels[sample_index], dtype=torch.long),
            "target_log_fee": torch.tensor(
                self.store.target_log_fee[sample_index], dtype=torch.float32
            ),
            "action_log_fees": torch.from_numpy(self.store.action_log_fees[sample_index]),
            "next_block_log_fee": torch.tensor(
                self.store.next_block_log_fee[sample_index], dtype=torch.float32
            ),
            "optimal_log_fee": torch.tensor(
                self.store.optimal_log_fee[sample_index], dtype=torch.float32
            ),
        }


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

"""LightningDataModule for temporal SPICE datasets."""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from ..data.datasets import TemporalDatasetStore
from ._runtime import build_sequence_loader
from .torch_datasets import SequenceBatch, build_class_weights

IntVector = NDArray[np.int64]


class TemporalDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        store: TemporalDatasetStore,
        train_sample_indices: IntVector,
        validation_sample_indices: IntVector,
        test_sample_indices: IntVector | None = None,
        predict_sample_indices: IntVector | None = None,
        lookback_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.store = store
        self.train_sample_indices = train_sample_indices
        self.validation_sample_indices = validation_sample_indices
        self.test_sample_indices = test_sample_indices
        self.predict_sample_indices = predict_sample_indices
        self.lookback_steps = lookback_steps
        self.batch_size = batch_size
        self.device = device
        self.class_weights = build_class_weights(
            store.class_labels,
            train_sample_indices,
            store.action_count,
        )

    def loader_for(self, sample_indices: IntVector) -> DataLoader[SequenceBatch]:
        return build_sequence_loader(
            self.store,
            sample_indices,
            lookback_steps=self.lookback_steps,
            batch_size=self.batch_size,
            device=self.device,
        )

    def train_dataloader(self) -> DataLoader[SequenceBatch]:
        return self.loader_for(self.train_sample_indices)

    def val_dataloader(self) -> DataLoader[SequenceBatch]:
        return self.loader_for(self.validation_sample_indices)

    def test_dataloader(self) -> DataLoader[SequenceBatch]:
        if self.test_sample_indices is None:
            raise RuntimeError("test_sample_indices were not configured")
        return self.loader_for(self.test_sample_indices)

    def predict_dataloader(self) -> DataLoader[SequenceBatch]:
        if self.predict_sample_indices is None:
            raise RuntimeError("predict_sample_indices were not configured")
        return self.loader_for(self.predict_sample_indices)

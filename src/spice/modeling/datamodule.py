"""LightningDataModule for temporal SPICE datasets."""

from __future__ import annotations

import lightning as L
import numpy as np
from numpy.typing import NDArray

from ..data.datasets import TemporalDatasetStore
from ._runtime import build_model_loader
from .representations import RepresentationLoader
from .torch_datasets import build_class_weights

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
        model_id: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.store = store
        self.train_sample_indices = train_sample_indices
        self.validation_sample_indices = validation_sample_indices
        self.test_sample_indices = test_sample_indices
        self.predict_sample_indices = predict_sample_indices
        self.model_id = model_id
        self.batch_size = batch_size
        self.class_weights = build_class_weights(
            store.class_labels,
            train_sample_indices,
            store.max_candidate_slots,
        )

    def loader_for(
        self,
        sample_indices: IntVector,
        *,
        shuffle: bool = False,
    ) -> RepresentationLoader:
        return build_model_loader(
            self.store,
            sample_indices,
            model_id=self.model_id,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> RepresentationLoader:
        return self.loader_for(self.train_sample_indices, shuffle=True)

    def val_dataloader(self) -> RepresentationLoader:
        return self.loader_for(self.validation_sample_indices)

    def test_dataloader(self) -> RepresentationLoader:
        if self.test_sample_indices is None:
            raise RuntimeError("test_sample_indices were not configured")
        return self.loader_for(self.test_sample_indices)

    def predict_dataloader(self) -> RepresentationLoader:
        if self.predict_sample_indices is None:
            raise RuntimeError("predict_sample_indices were not configured")
        return self.loader_for(self.predict_sample_indices)

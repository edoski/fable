"""LightningDataModule for temporal SPICE datasets."""

from __future__ import annotations

import lightning as L

from ..prediction import PredictionBatch
from .batch_sources import PreparedBatchSource


class TemporalDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_batch_source: PreparedBatchSource,
        validation_batch_source: PreparedBatchSource,
        test_batch_source: PreparedBatchSource | None = None,
        predict_batch_source: PreparedBatchSource | None = None,
    ) -> None:
        super().__init__()
        self._train_loader = train_batch_source
        self._validation_loader = validation_batch_source
        self._test_loader = test_batch_source
        self._predict_loader = (
            None
            if predict_batch_source is None
            else predict_batch_source
        )

    def train_dataloader(self) -> PreparedBatchSource:
        return self._train_loader

    def val_dataloader(self) -> PreparedBatchSource:
        return self._validation_loader

    def test_dataloader(self) -> PreparedBatchSource:
        if self._test_loader is None:
            raise RuntimeError("test_batch_source was not configured")
        return self._test_loader

    def predict_dataloader(self) -> PreparedBatchSource:
        if self._predict_loader is None:
            raise RuntimeError("predict_batch_source was not configured")
        return self._predict_loader

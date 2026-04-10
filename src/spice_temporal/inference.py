"""Inference helpers for trained temporal models."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from spice_temporal._runtime import build_sequence_loader, resolve_device
from spice_temporal.datasets import TemporalDatasetStore
from spice_temporal.models import TemporalModel

IntVector = NDArray[np.int64]


def predict_class_offsets(
    model: TemporalModel,
    *,
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    lookback_steps: int,
    effective_batch_size: int,
    device: str,
) -> list[int]:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    loader = build_sequence_loader(
        store,
        sample_indices,
        lookback_steps=lookback_steps,
        effective_batch_size=effective_batch_size,
        device=resolved_device,
    )

    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["inputs"].to(resolved_device)).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return predictions

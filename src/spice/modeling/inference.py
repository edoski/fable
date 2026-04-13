"""Inference helpers for trained temporal models."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.console import NullReporter, Reporter
from ..data.datasets import TemporalDatasetStore
from ._runtime import build_model_loader, resolve_device
from .models import TemporalModel
from .torch_datasets import move_batch_to_device

IntVector = NDArray[np.int64]


def predict_class_offsets(
    model: TemporalModel,
    *,
    model_id: str,
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    batch_size: int,
    device: str,
    reporter: Reporter | None = None,
) -> list[int]:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    loader = build_model_loader(
        store,
        sample_indices,
        model_id=model_id,
        batch_size=batch_size,
    )
    task_id = reporter.start_task("predict offsets", total=len(loader), unit="batches")
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = move_batch_to_device(batch, resolved_device)
            logits = model(device_batch.inputs, device_batch.input_mask).logits
            logits = logits.masked_fill(
                ~device_batch.candidate_mask,
                torch.finfo(logits.dtype).min,
            )
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            reporter.update_task(task_id, advance=1)
    reporter.finish_task(task_id)
    return predictions

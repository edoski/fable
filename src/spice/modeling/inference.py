"""Inference helpers for trained temporal models."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.reporting import NullReporter, Reporter
from ..prediction import CompiledPredictionContract
from ..temporal.problem_store import CompiledProblemStore
from ._runtime import (
    CompiledRepresentationContract,
    build_prediction_loader,
    build_representation_runtime_context,
    resolve_device,
)
from .models import TemporalModel

IntVector = NDArray[np.int64]


def predict_with_model(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    batch_size: int,
    device: str,
    reporter: Reporter | None = None,
) -> object:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    runtime_context = build_representation_runtime_context(
        device=resolved_device,
        batch_size=batch_size,
    )
    loader = build_prediction_loader(
        store,
        sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        runtime_context=runtime_context,
        seed=0,
    )
    task_id = reporter.start_task("predict", total=len(loader), unit="batches")
    predictions = prediction_contract.allocate_decoded_offsets(int(sample_indices.shape[0]))
    with torch.no_grad():
        for batch in loader:
            device_batch = batch.to_device(resolved_device)
            outputs = model(**device_batch.model_kwargs())
            prediction_contract.decode_selected_offsets_into(
                predictions,
                batch.sample_positions,
                outputs,
                device_batch.targets,
            )
            reporter.update_task(task_id, advance=1)
    reporter.finish_task(task_id)
    return predictions

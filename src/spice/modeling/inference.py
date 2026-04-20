"""Inference helpers for trained temporal models."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.reporting import NullReporter, Reporter
from ..prediction import CompiledPredictionContract, DecodedOffsets
from ..temporal.problem_store import CompiledProblemStore
from ._runtime import (
    CompiledRepresentationContract,
    build_inference_batch_source,
    build_representation_runtime_context,
    configure_torch_backends,
    ensure_device_runtime_ready,
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
) -> DecodedOffsets:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    resolved_device = resolve_device(device)
    ensure_device_runtime_ready(
        requested_device=device,
        resolved_device=resolved_device,
    )
    model.to(resolved_device)
    model.eval()
    runtime_context = build_representation_runtime_context(
        device=resolved_device,
        batch_size=batch_size,
    )
    batch_source_plan = build_inference_batch_source(
        store,
        sample_indices,
        representation_contract=representation_contract,
        runtime_context=runtime_context,
        resolved_device=resolved_device,
        seed=0,
    )
    loader = batch_source_plan.source
    task_id = reporter.start_task("predict", total=len(loader), unit="batches")
    predictions = prediction_contract.allocate_decoded_offsets(int(sample_indices.shape[0]))
    with configure_torch_backends(
        resolved_device=resolved_device,
        deterministic=None,
    ):
        with torch.no_grad():
            for batch in loader:
                device_batch = batch.to_device(resolved_device)
                outputs = model(**device_batch.model_kwargs())
                prediction_contract.decode_selected_offsets_into(
                    predictions,
                    batch.sample_positions,
                    outputs,
                )
                reporter.update_task(task_id, advance=1)
    reporter.finish_task(task_id)
    return predictions

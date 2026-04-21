"""Inference helpers for trained temporal models."""

from __future__ import annotations

from ..prediction import (
    CompiledPredictionContract,
    DecodedOffsets,
    decode_context_from_batch,
)
from ..temporal.problem_store import CompiledProblemStore, IntVector
from ._runtime import (
    build_cuda_modeling_runtime,
    configure_torch_backends,
    run_model_forward_pass,
)
from .batch_sources import build_model_input_batch_source
from .families.base import ModelConfig
from .families.registry import resolve_model_training_precision
from .models import TemporalModel
from .representations import CompiledRepresentationContract


def predict_with_model(
    model: TemporalModel,
    *,
    model_config: ModelConfig,
    prediction_contract: CompiledPredictionContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    batch_size: int,
) -> DecodedOffsets:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    runtime = build_cuda_modeling_runtime(batch_size=batch_size)
    precision = resolve_model_training_precision(
        device=runtime.resolved_device,
        model_config=model_config,
    )
    model.to(runtime.resolved_device)
    batch_source = build_model_input_batch_source(
        store,
        sample_indices,
        representation_contract=representation_contract,
        runtime_context=runtime.representation_runtime_context,
        resolved_device=runtime.resolved_device,
        seed=0,
    )
    predictions = prediction_contract.allocate_decoded_offsets(int(sample_indices.shape[0]))

    def _decode_batch(batch, outputs) -> None:
        prediction_contract.decode_selected_offsets_into(
            predictions,
            outputs,
            decode_context_from_batch(batch),
        )

    with configure_torch_backends(
        resolved_device=runtime.resolved_device,
        deterministic=None,
    ):
        run_model_forward_pass(
            model,
            loader=batch_source,
            resolved_device=runtime.resolved_device,
            precision=precision,
            on_outputs=_decode_batch,
        )
    return predictions

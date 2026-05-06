"""Inference helpers for trained temporal models."""

from __future__ import annotations

import torch

from ..core.errors import SpiceOperatorError
from ..prediction import (
    CompiledPredictionContract,
)
from ..prediction.decoding import DecodedPredictionResult, decode_context_from_batch
from ..temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ..temporal.problem_store import CompiledProblemStore
from .forward_runtime import run_planned_model_input_forward
from .models import TemporalModel
from .representations import CompiledRepresentationContract
from .runtime_planning import (
    ModelingRuntimePlan,
    modeling_backend_scope,
    prepare_model_for_runtime,
)


def predict_with_model(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    representation_contract: CompiledRepresentationContract,
    execution_policy: CompiledExecutionPolicyContract,
    store: CompiledProblemStore,
    action_space: PreparedActionSpace,
    runtime_plan: ModelingRuntimePlan,
) -> DecodedPredictionResult:
    if action_space.sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    runtime_model = prepare_model_for_runtime(model, runtime_plan)
    predictions = prediction_contract.allocate_decoded_result(
        int(action_space.sample_indices.shape[0])
    )

    def _decode_batch(batch, outputs) -> None:
        _validate_finite_outputs(outputs)
        prediction_contract.decode_batch_result_into(
            predictions,
            outputs,
            decode_context_from_batch(batch),
        )

    with modeling_backend_scope(runtime_plan):
        run_planned_model_input_forward(
            runtime_model,
            store=store,
            action_space=action_space,
            representation_contract=representation_contract,
            execution_policy=execution_policy,
            runtime_plan=runtime_plan,
            on_outputs=_decode_batch,
        )
    return predictions


def _validate_finite_outputs(outputs) -> None:
    for head_id, tensor in outputs.heads.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise SpiceOperatorError(f"Non-finite model output head during inference: {head_id}")

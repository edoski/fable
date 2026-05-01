"""Training runtime planning and CUDA memory probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from ..config.models import TrainingConfig
from ..prediction import CompiledPredictionContract
from ..prediction.contracts import PredictionBatch
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from ._epoch_execution import execute_training_batch
from ._runtime import (
    compute_device_resident_budget,
    peak_cuda_reserved_bytes,
    reset_cuda_peak_memory,
    snapshot_cuda_memory,
)
from .batch_plan import BatchPlan, build_prediction_batch_plan
from .evaluation_runtime import EvaluationScoringRuntimePlan
from .models import TemporalModel
from .representations import CompiledRepresentationContract, RepresentationRuntimeContext


@dataclass(frozen=True, slots=True)
class TrainingRuntimePlan:
    runtime_context: RepresentationRuntimeContext
    train_batch_plan: BatchPlan[PredictionBatch]
    validation_batch_plan: BatchPlan[PredictionBatch]
    evaluation_scoring_runtime_plan: EvaluationScoringRuntimePlan
    prediction_training_state: object | None


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


def _clone_cpu_state(model: TemporalModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _host_warmup_context(
    runtime_context: RepresentationRuntimeContext,
) -> RepresentationRuntimeContext:
    return runtime_context.with_device_memory_budget(0)


def plan_training_runtime(
    model: TemporalModel,
    *,
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    representation_contract: CompiledRepresentationContract,
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    training_config: TrainingConfig,
    precision: str,
) -> TrainingRuntimePlan:
    warmup_plan = build_prediction_batch_plan(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=_host_warmup_context(base_runtime_context),
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=False,
    )
    warmup_state = _clone_cpu_state(_unwrap_compiled_model(model))
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    prediction_training_state = prediction_contract.fit_training_state(
        store,
        train_sample_indices,
        execution_policy=execution_policy,
    )
    budget = 0
    try:
        baseline_memory = snapshot_cuda_memory(resolved_device)
        reset_cuda_peak_memory(resolved_device)
        batch = next(iter(warmup_plan.source))
        execute_training_batch(
            model,
            batch,
            resolved_device=resolved_device,
            precision=precision,
            prediction_contract=prediction_contract,
            prediction_training_state=prediction_training_state,
            optimizer=warmup_optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            zero_after_step=True,
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        budget = compute_device_resident_budget(
            free_bytes=baseline_memory.free_bytes,
            baseline_reserved_bytes=baseline_memory.reserved_bytes,
            peak_reserved_bytes=peak_cuda_reserved_bytes(resolved_device),
            total_bytes=baseline_memory.total_bytes,
        )
    finally:
        _unwrap_compiled_model(model).load_state_dict(warmup_state)
        del warmup_plan, warmup_optimizer
        torch.cuda.empty_cache()
    planned_runtime_context = base_runtime_context.with_device_memory_budget(budget)
    train_batch_plan = build_prediction_batch_plan(
        store,
        train_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=True,
    )
    validation_batch_plan = build_prediction_batch_plan(
        store,
        validation_sample_indices,
        representation_contract=representation_contract,
        prediction_contract=prediction_contract,
        execution_policy=execution_policy,
        runtime_context=planned_runtime_context,
        resolved_device=resolved_device,
        seed=training_config.seed,
        shuffle=False,
    )
    return TrainingRuntimePlan(
        runtime_context=planned_runtime_context,
        train_batch_plan=train_batch_plan,
        validation_batch_plan=validation_batch_plan,
        evaluation_scoring_runtime_plan=EvaluationScoringRuntimePlan(
            resolved_device=resolved_device,
            precision=precision,
            representation_runtime_context=planned_runtime_context,
            deterministic=None,
            seed=0,
        ),
        prediction_training_state=prediction_training_state,
    )

"""Planned forward-only runtime execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from ..prediction import CompiledPredictionContract
from ..prediction.contracts import ModelInputBatch, PredictionBatch
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from ._runtime import measure_forward_device_resident_budget, run_model_forward_pass
from .batch_plan import BatchPlan, build_model_input_batch_plan, build_prediction_batch_plan
from .models import ModelOutputs, TemporalModel
from .representations import CompiledRepresentationContract, RepresentationRuntimeContext

ForwardBatchT = TypeVar("ForwardBatchT", ModelInputBatch, PredictionBatch)


def _host_warmup_context(
    runtime_context: RepresentationRuntimeContext,
) -> RepresentationRuntimeContext:
    return runtime_context.with_device_memory_budget(0)


def _require_non_empty_samples(sample_indices: IntVector) -> None:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")


def _run_planned_forward(
    model: TemporalModel,
    *,
    build_plan: Callable[[RepresentationRuntimeContext], BatchPlan[ForwardBatchT]],
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    precision: str,
    on_outputs: Callable[[ForwardBatchT, ModelOutputs], None],
) -> None:
    warmup_plan = build_plan(_host_warmup_context(base_runtime_context))
    budget = measure_forward_device_resident_budget(
        model,
        loader=warmup_plan.source,
        resolved_device=resolved_device,
        precision=precision,
    )
    planned_runtime_context = base_runtime_context.with_device_memory_budget(budget)
    del warmup_plan
    batch_plan = build_plan(planned_runtime_context)
    run_model_forward_pass(
        model,
        loader=batch_plan.source,
        resolved_device=resolved_device,
        precision=precision,
        on_outputs=on_outputs,
    )


def run_planned_model_input_forward(
    model: TemporalModel,
    *,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    representation_contract: CompiledRepresentationContract,
    execution_policy: CompiledExecutionPolicyContract,
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    precision: str,
    seed: int,
    on_outputs: Callable[[ModelInputBatch, ModelOutputs], None],
) -> None:
    _require_non_empty_samples(sample_indices)

    def _build_plan(runtime_context: RepresentationRuntimeContext) -> BatchPlan[ModelInputBatch]:
        return build_model_input_batch_plan(
            store,
            sample_indices,
            representation_contract=representation_contract,
            execution_policy=execution_policy,
            runtime_context=runtime_context,
            resolved_device=resolved_device,
            seed=seed,
        )

    _run_planned_forward(
        model,
        build_plan=_build_plan,
        base_runtime_context=base_runtime_context,
        resolved_device=resolved_device,
        precision=precision,
        on_outputs=on_outputs,
    )


def run_planned_prediction_forward(
    model: TemporalModel,
    *,
    store: CompiledProblemStore,
    sample_indices: IntVector,
    representation_contract: CompiledRepresentationContract,
    prediction_contract: CompiledPredictionContract,
    execution_policy: CompiledExecutionPolicyContract,
    base_runtime_context: RepresentationRuntimeContext,
    resolved_device: torch.device,
    precision: str,
    seed: int,
    on_outputs: Callable[[PredictionBatch, ModelOutputs], None],
) -> None:
    _require_non_empty_samples(sample_indices)

    def _build_plan(runtime_context: RepresentationRuntimeContext) -> BatchPlan[PredictionBatch]:
        return build_prediction_batch_plan(
            store,
            sample_indices,
            representation_contract=representation_contract,
            prediction_contract=prediction_contract,
            execution_policy=execution_policy,
            runtime_context=runtime_context,
            resolved_device=resolved_device,
            seed=seed,
            shuffle=False,
        )

    _run_planned_forward(
        model,
        build_plan=_build_plan,
        base_runtime_context=base_runtime_context,
        resolved_device=resolved_device,
        precision=precision,
        on_outputs=on_outputs,
    )

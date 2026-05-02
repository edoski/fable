"""Planned forward-only runtime execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from ..prediction import CompiledPredictionContract
from ..prediction.contracts import ModelInputBatch, PredictionBatch
from ..temporal.execution_policy import CompiledExecutionPolicyContract
from ..temporal.problem_store import CompiledProblemStore, IntVector
from ._runtime import ForwardBatch, precision_context, run_model_forward_pass
from ._runtime_probe import measure_device_resident_budget, measured_runtime_context
from .batch_plan import (
    BatchPlan,
    BatchSource,
    build_model_input_batch_plan,
    build_prediction_batch_plan,
)
from .models import ModelOutputs, TemporalModel
from .representations import CompiledRepresentationContract, RepresentationRuntimeContext

ForwardBatchT = TypeVar("ForwardBatchT", ModelInputBatch, PredictionBatch)


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
    planned_runtime_context = measured_runtime_context(
        base_runtime_context,
        build_warmup_plan=build_plan,
        measure_warmup_budget=lambda warmup_plan: _measure_forward_batch_budget(
            model,
            loader=warmup_plan.source,
            resolved_device=resolved_device,
            precision=precision,
        ),
    )
    batch_plan = build_plan(planned_runtime_context)
    run_model_forward_pass(
        model,
        loader=batch_plan.source,
        resolved_device=resolved_device,
        precision=precision,
        on_outputs=on_outputs,
    )


def _measure_forward_batch_budget(
    model: TemporalModel,
    *,
    loader: BatchSource[ForwardBatch],
    resolved_device: torch.device,
    precision: str,
) -> int:
    def _run_forward_probe() -> None:
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            device_batch = batch.to_device(resolved_device)
            with precision_context(precision=precision):
                _ = model(**device_batch.model_kwargs())

    return measure_device_resident_budget(
        resolved_device=resolved_device,
        run_probe=_run_forward_probe,
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

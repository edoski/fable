"""Private runtime budget probe helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from ._runtime import (
    compute_device_resident_budget,
    peak_cuda_reserved_bytes,
    reset_cuda_peak_memory,
    snapshot_cuda_memory,
)
from .representations import DeviceStorageBudget, RepresentationRuntimeContext
from .runtime_planning import ModelingRuntimePlan, with_representation_runtime_context

WarmupPlanT = TypeVar("WarmupPlanT")


def _host_warmup_context(
    runtime_context: RepresentationRuntimeContext,
) -> RepresentationRuntimeContext:
    return runtime_context.with_device_storage_budget(
        DeviceStorageBudget.disabled()
    ).with_host_loader_policy("single_process_unpinned")


def build_measured_modeling_runtime_plan(
    base_runtime_plan: ModelingRuntimePlan,
    *,
    build_warmup_plan: Callable[[ModelingRuntimePlan], WarmupPlanT],
    measure_warmup_budget: Callable[[WarmupPlanT, ModelingRuntimePlan], int],
) -> ModelingRuntimePlan:
    warmup_runtime_plan = with_representation_runtime_context(
        base_runtime_plan,
        _host_warmup_context(base_runtime_plan.representation_runtime_context),
    )
    warmup_plan = build_warmup_plan(warmup_runtime_plan)
    budget = measure_warmup_budget(warmup_plan, warmup_runtime_plan)
    measured_context = base_runtime_plan.representation_runtime_context.with_device_storage_budget(
        DeviceStorageBudget.measured(budget)
    )
    return with_representation_runtime_context(base_runtime_plan, measured_context)


def measure_device_resident_budget(
    *,
    resolved_device: torch.device,
    run_probe: Callable[[], None],
) -> int:
    baseline = snapshot_cuda_memory(resolved_device)
    reset_cuda_peak_memory(resolved_device)
    run_probe()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    return compute_device_resident_budget(
        free_bytes=baseline.free_bytes,
        baseline_reserved_bytes=baseline.reserved_bytes,
        peak_reserved_bytes=peak_cuda_reserved_bytes(resolved_device),
        total_bytes=baseline.total_bytes,
    )

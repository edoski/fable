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

WarmupPlanT = TypeVar("WarmupPlanT")


def host_warmup_context(
    runtime_context: RepresentationRuntimeContext,
) -> RepresentationRuntimeContext:
    return runtime_context.with_device_storage_budget(DeviceStorageBudget.disabled())


def measured_runtime_context(
    base_runtime_context: RepresentationRuntimeContext,
    *,
    build_warmup_plan: Callable[[RepresentationRuntimeContext], WarmupPlanT],
    measure_warmup_budget: Callable[[WarmupPlanT], int],
) -> RepresentationRuntimeContext:
    warmup_plan = build_warmup_plan(host_warmup_context(base_runtime_context))
    budget = measure_warmup_budget(warmup_plan)
    return base_runtime_context.with_device_storage_budget(
        DeviceStorageBudget.measured(budget)
    )


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

from __future__ import annotations

import pytest
import torch

from spice.modeling._runtime import CudaMemorySnapshot
from spice.modeling._runtime_probe import (
    host_warmup_context,
    measure_device_resident_budget,
    measured_runtime_context,
)
from spice.modeling.representations import DeviceStorageBudget, RepresentationRuntimeContext


def test_device_storage_budget_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="bytes must be non-negative"):
        DeviceStorageBudget.measured(-1)
    with pytest.raises(ValueError, match="disabled device storage budget"):
        DeviceStorageBudget(phase="disabled", bytes=1)
    with pytest.raises(ValueError, match="measured device storage budget"):
        DeviceStorageBudget(phase="measured", bytes=None)
    with pytest.raises(ValueError, match="phase is unsupported"):
        DeviceStorageBudget(phase="legacy", bytes=1)  # type: ignore[arg-type]


def test_measured_runtime_context_uses_host_warmup_then_measured_budget() -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=4,
        available_host_memory_bytes=1024,
        device_storage_budget=DeviceStorageBudget.coarse(999),
    )
    seen_budgets: list[DeviceStorageBudget] = []

    def build_warmup_plan(context: RepresentationRuntimeContext) -> object:
        seen_budgets.append(context.device_storage_budget)
        return object()

    planned_context = measured_runtime_context(
        runtime_context,
        build_warmup_plan=build_warmup_plan,
        measure_warmup_budget=lambda _plan: 123,
    )

    assert host_warmup_context(runtime_context).device_storage_budget == (
        DeviceStorageBudget.disabled()
    )
    assert seen_budgets == [DeviceStorageBudget.disabled()]
    assert planned_context.device_storage_budget == DeviceStorageBudget.measured(123)


def test_measure_device_resident_budget_runs_probe_inside_cuda_accounting(
    monkeypatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        "spice.modeling._runtime_probe.snapshot_cuda_memory",
        lambda _device: events.append("snapshot")
        or CudaMemorySnapshot(
            free_bytes=1000,
            total_bytes=10000,
            allocated_bytes=10,
            reserved_bytes=100,
        ),
    )
    monkeypatch.setattr(
        "spice.modeling._runtime_probe.reset_cuda_peak_memory",
        lambda _device: events.append("reset"),
    )
    monkeypatch.setattr(
        "spice.modeling._runtime_probe.peak_cuda_reserved_bytes",
        lambda _device: events.append("peak") or 200,
    )
    monkeypatch.setattr(
        "spice.modeling._runtime_probe.compute_device_resident_budget",
        lambda **kwargs: events.append("compute")
        or kwargs["free_bytes"]
        - (kwargs["peak_reserved_bytes"] - kwargs["baseline_reserved_bytes"]),
    )
    monkeypatch.setattr(
        "spice.modeling._runtime_probe.torch.cuda.synchronize",
        lambda _device: events.append("sync"),
    )

    budget = measure_device_resident_budget(
        resolved_device=torch.device("cuda"),
        run_probe=lambda: events.append("probe"),
    )

    assert budget == 900
    assert events == ["snapshot", "reset", "probe", "sync", "peak", "compute"]

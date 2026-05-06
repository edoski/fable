from __future__ import annotations

import pytest
import torch

from spice.modeling._runtime import CudaMemorySnapshot
from spice.modeling._runtime_probe import (
    build_measured_modeling_runtime_plan,
    measure_device_resident_budget,
)
from spice.modeling.representations import DeviceStorageBudget, RepresentationRuntimeContext
from spice.modeling.runtime_planning import ModelingRuntimePlan


def test_device_storage_budget_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="bytes must be non-negative"):
        DeviceStorageBudget.measured(-1)
    with pytest.raises(ValueError, match="disabled device storage budget"):
        DeviceStorageBudget(phase="disabled", bytes=1)
    with pytest.raises(ValueError, match="measured device storage budget"):
        DeviceStorageBudget(phase="measured", bytes=None)
    with pytest.raises(ValueError, match="phase is unsupported"):
        DeviceStorageBudget(phase="unknown", bytes=1)  # type: ignore[arg-type]


def test_build_measured_modeling_runtime_plan_uses_host_warmup_then_final_plan() -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=4,
        available_host_memory_bytes=1024,
        device_storage_budget=DeviceStorageBudget.coarse(999),
        host_loader_policy="automatic",
    )
    runtime_plan = ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=runtime_context,
        deterministic=True,
        seed=42,
        compile_enabled=True,
    )
    seen_warmup_plans: list[ModelingRuntimePlan] = []
    seen_measure_plans: list[ModelingRuntimePlan] = []

    def build_warmup_plan(plan: ModelingRuntimePlan) -> object:
        seen_warmup_plans.append(plan)
        return object()

    def measure_warmup_budget(_plan: object, warmup_plan: ModelingRuntimePlan) -> int:
        seen_measure_plans.append(warmup_plan)
        return 123

    measured_plan = build_measured_modeling_runtime_plan(
        runtime_plan,
        build_warmup_plan=build_warmup_plan,
        measure_warmup_budget=measure_warmup_budget,
    )

    warmup_plan = seen_warmup_plans[0]
    assert seen_measure_plans == [warmup_plan]
    assert warmup_plan.representation_runtime_context.device_storage_budget == (
        DeviceStorageBudget.disabled()
    )
    assert warmup_plan.representation_runtime_context.host_loader_policy == (
        "single_process_unpinned"
    )
    assert measured_plan.representation_runtime_context.device_storage_budget == (
        DeviceStorageBudget.measured(123)
    )
    assert measured_plan.representation_runtime_context.host_loader_policy == "automatic"
    assert measured_plan.resolved_device == runtime_plan.resolved_device
    assert measured_plan.precision == runtime_plan.precision
    assert measured_plan.deterministic == runtime_plan.deterministic
    assert measured_plan.seed == runtime_plan.seed
    assert measured_plan.compile_enabled == runtime_plan.compile_enabled


def test_measure_device_resident_budget_runs_probe_inside_cuda_accounting(
    monkeypatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        "spice.modeling._runtime_probe.snapshot_cuda_memory",
        lambda _device: (
            events.append("snapshot")
            or CudaMemorySnapshot(
                free_bytes=1000,
                total_bytes=10000,
                allocated_bytes=10,
                reserved_bytes=100,
            )
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
        lambda **kwargs: (
            events.append("compute")
            or kwargs["free_bytes"]
            - (kwargs["peak_reserved_bytes"] - kwargs["baseline_reserved_bytes"])
        ),
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

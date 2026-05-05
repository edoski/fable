from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from spice.core.errors import SpiceOperatorError
from spice.modeling._runtime import (
    CudaModelingRuntime,
    build_representation_runtime_context,
    compute_device_resident_budget,
    default_device_resident_safety_margin,
    ensure_cuda_runtime_ready,
    resolve_coarse_device_storage_budget,
)
from spice.modeling.representations import DeviceStorageBudget
from spice.modeling.runtime_planning import build_cuda_modeling_runtime_plan


def test_ensure_cuda_runtime_ready_raises_clear_error_for_broken_cuda_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("driver mismatch")),
    )

    with pytest.raises(SpiceOperatorError, match="CUDA runtime initialization failed"):
        ensure_cuda_runtime_ready(torch.device("cuda"))


def test_ensure_cuda_runtime_ready_rejects_non_cuda_resolutions() -> None:
    with pytest.raises(SpiceOperatorError, match="Modeling runtime requires CUDA devices"):
        ensure_cuda_runtime_ready(torch.device("cpu"))


def test_ensure_cuda_runtime_ready_rejects_rocm(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)

    with pytest.raises(SpiceOperatorError, match="ROCm/HIP is unsupported"):
        ensure_cuda_runtime_ready(torch.device("cuda"))


def test_coarse_device_storage_budget_returns_none_for_cpu() -> None:
    assert resolve_coarse_device_storage_budget(torch.device("cpu")) is None


def test_coarse_device_storage_budget_uses_current_cuda_device(monkeypatch) -> None:
    seen_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    def fake_mem_get_info(device):
        seen_devices.append(device)
        return 1_000, 2_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    assert resolve_coarse_device_storage_budget(torch.device("cuda")) == 500
    assert seen_devices == [2]


def test_coarse_device_storage_budget_uses_explicit_cuda_device(monkeypatch) -> None:
    seen_devices: list[int] = []

    def fake_mem_get_info(device):
        seen_devices.append(device)
        return 2_000, 4_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    assert resolve_coarse_device_storage_budget(torch.device("cuda:3")) == 1_000
    assert seen_devices == [3]


def test_representation_runtime_context_wraps_coarse_device_storage_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr("spice.modeling._runtime._available_system_memory_bytes", lambda: 4096)
    monkeypatch.setattr(
        "spice.modeling._runtime.resolve_coarse_device_storage_budget",
        lambda _device: 123,
    )

    context = build_representation_runtime_context(
        device=torch.device("cuda"),
        batch_size=8,
    )

    assert context.batch_size == 8
    assert context.available_host_memory_bytes == 4096
    assert context.device_storage_budget == DeviceStorageBudget.coarse(123)


def test_cuda_modeling_runtime_plan_owns_precision_and_backend_facts(monkeypatch) -> None:
    runtime_context = build_representation_runtime_context(
        device=torch.device("cpu"),
        batch_size=8,
    )
    monkeypatch.setattr(
        "spice.modeling.runtime_planning.build_cuda_modeling_runtime",
        lambda **_: CudaModelingRuntime(
            resolved_device=torch.device("cpu"),
            representation_runtime_context=runtime_context,
        ),
    )

    plan = build_cuda_modeling_runtime_plan(
        model_config=cast(Any, object()),
        batch_size=8,
        deterministic=True,
        seed=17,
    )

    assert plan.resolved_device == torch.device("cpu")
    assert plan.precision == "32-true"
    assert plan.representation_runtime_context is runtime_context
    assert plan.deterministic is True
    assert plan.seed == 17
    assert plan.compile_enabled is False


def test_default_device_resident_safety_margin_uses_five_percent_floor() -> None:
    total_bytes = 44 * 1024**3

    margin = default_device_resident_safety_margin(total_bytes)

    assert margin == total_bytes // 20


def test_compute_device_resident_budget_subtracts_peak_increment_and_margin() -> None:
    free_bytes = 37 * 1024**3
    baseline_reserved_bytes = 4 * 1024**3
    peak_reserved_bytes = 12 * 1024**3
    total_bytes = 44 * 1024**3

    budget = compute_device_resident_budget(
        free_bytes=free_bytes,
        baseline_reserved_bytes=baseline_reserved_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        total_bytes=total_bytes,
    )

    assert budget == free_bytes - (peak_reserved_bytes - baseline_reserved_bytes) - (
        total_bytes // 20
    )


def test_compute_device_resident_budget_clamps_to_zero() -> None:
    budget = compute_device_resident_budget(
        free_bytes=1024,
        baseline_reserved_bytes=2048,
        peak_reserved_bytes=4096,
        total_bytes=1024**3,
    )

    assert budget == 0

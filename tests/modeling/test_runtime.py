from __future__ import annotations

import pytest
import torch

from spice.core.errors import SpiceOperatorError
from spice.modeling._runtime import (
    compute_device_resident_budget,
    default_device_resident_safety_margin,
    ensure_cuda_runtime_ready,
    resolve_available_device_memory_budget,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.registry import (
    resolve_model_compile_enabled,
    resolve_model_training_precision,
)
from spice.modeling.families.transformer import TransformerModelConfig
from spice.modeling.families.transformer_lstm import TransformerLstmModelConfig


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


def test_available_device_memory_budget_returns_none_for_cpu() -> None:
    assert resolve_available_device_memory_budget(torch.device("cpu")) is None


def test_available_device_memory_budget_uses_current_cuda_device(monkeypatch) -> None:
    seen_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    def fake_mem_get_info(device):
        seen_devices.append(device)
        return 1_000, 2_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    assert resolve_available_device_memory_budget(torch.device("cuda")) == 500
    assert seen_devices == [2]


def test_available_device_memory_budget_uses_explicit_cuda_device(monkeypatch) -> None:
    seen_devices: list[int] = []

    def fake_mem_get_info(device):
        seen_devices.append(device)
        return 2_000, 4_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    assert resolve_available_device_memory_budget(torch.device("cuda:3")) == 1_000
    assert seen_devices == [3]


def test_transformer_disables_auto_compile_on_cuda() -> None:
    enabled = resolve_model_compile_enabled(
        device=torch.device("cuda"),
        model_config=TransformerModelConfig(
            dropout=0.1,
            d_model=16,
            nhead=4,
            transformer_layers=2,
            feedforward_dim=32,
            head_hidden_dim=8,
        ),
    )

    assert enabled is False


def test_recurrent_families_disable_auto_compile_on_cuda() -> None:
    lstm_enabled = resolve_model_compile_enabled(
        device=torch.device("cuda"),
        model_config=LstmModelConfig(
            input_projection_dim=8,
            hidden_size=16,
            num_layers=2,
            dropout=0.1,
            head_hidden_dim=8,
        ),
    )
    transformer_lstm_enabled = resolve_model_compile_enabled(
        device=torch.device("cuda"),
        model_config=TransformerLstmModelConfig(
            hidden_size=16,
            num_layers=2,
            dropout=0.1,
            d_model=16,
            nhead=4,
            transformer_layers=2,
            feedforward_dim=32,
            head_hidden_dim=8,
        ),
    )

    assert lstm_enabled is False
    assert transformer_lstm_enabled is False


def test_recurrent_families_default_to_fp32_on_cuda() -> None:
    assert (
        resolve_model_training_precision(
            device=torch.device("cuda"),
            model_config=LstmModelConfig(
                input_projection_dim=8,
                hidden_size=16,
                num_layers=2,
                dropout=0.1,
                head_hidden_dim=8,
            ),
        )
        == "32-true"
    )
    assert (
        resolve_model_training_precision(
            device=torch.device("cuda"),
            model_config=TransformerLstmModelConfig(
                hidden_size=16,
                num_layers=2,
                dropout=0.1,
                d_model=16,
                nhead=4,
                transformer_layers=2,
                feedforward_dim=32,
                head_hidden_dim=8,
            ),
        )
        == "32-true"
    )


def test_transformer_defaults_to_fp32_on_cuda() -> None:
    assert (
        resolve_model_training_precision(
            device=torch.device("cuda"),
            model_config=TransformerModelConfig(
                dropout=0.1,
                d_model=16,
                nhead=4,
                transformer_layers=2,
                feedforward_dim=32,
                head_hidden_dim=8,
            ),
        )
        == "32-true"
    )


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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spice.core.errors import SpiceOperatorError
from spice.modeling._runtime import (
    compute_device_resident_budget,
    default_device_resident_safety_margin,
    ensure_cuda_runtime_ready,
    resolve_cuda_device,
)
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.registry import (
    resolve_model_compile_enabled,
    resolve_model_training_precision,
)
from spice.modeling.families.transformer import TransformerModelConfig
from spice.modeling.families.transformer_lstm import TransformerLstmModelConfig


def test_resolve_cuda_device_returns_cuda() -> None:
    assert resolve_cuda_device() == torch.device("cuda")


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


def test_resolve_compile_enabled_skips_transformer_auto_compile_on_small_cuda_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=32),
    )

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


def test_resolve_compile_enabled_keeps_transformer_auto_compile_on_big_cuda_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=72),
    )

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

    assert enabled is True


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


def test_transformer_default_precision_prefers_bf16_on_supported_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

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
        == "bf16-mixed"
    )


def test_transformer_default_precision_falls_back_to_fp16_without_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

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
        == "16-mixed"
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

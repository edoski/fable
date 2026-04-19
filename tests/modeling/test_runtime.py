from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spice.config import CompileMode, TrainingPrecision
from spice.core.errors import SpiceOperatorError
from spice.modeling._runtime import ensure_device_runtime_ready, resolve_compile_enabled
from spice.modeling.families.lstm import LstmModelConfig
from spice.modeling.families.registry import resolve_default_precision
from spice.modeling.families.transformer import TransformerModelConfig
from spice.modeling.families.transformer_lstm import TransformerLstmModelConfig


def test_resolve_compile_enabled_skips_auto_compile_on_small_cuda_gpu(
    tmp_path,
    monkeypatch,
    load_test_train_config,
    model_workflow_override,
) -> None:
    config = load_test_train_config(tmp_path, override=model_workflow_override())

    monkeypatch.setattr("spice.modeling._runtime.resolve_auto_compile", lambda *args: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=32),
    )

    training = config.training.model_copy(update={"compile": CompileMode.AUTO})
    enabled = resolve_compile_enabled(
        training,
        device=torch.device("cuda"),
        precision="bf16-mixed",
        model_config=config.model,
    )

    assert enabled is False


def test_resolve_compile_enabled_keeps_explicit_compile_on_for_small_cuda_gpu(
    tmp_path,
    monkeypatch,
    load_test_train_config,
    model_workflow_override,
) -> None:
    config = load_test_train_config(tmp_path, override=model_workflow_override())

    monkeypatch.setattr("spice.modeling._runtime.resolve_auto_compile", lambda *args: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=32),
    )

    training = config.training.model_copy(update={"compile": CompileMode.ON})
    enabled = resolve_compile_enabled(
        training,
        device=torch.device("cuda"),
        precision="bf16-mixed",
        model_config=config.model,
    )

    assert enabled is True


def test_ensure_device_runtime_ready_raises_clear_error_for_broken_cuda_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("driver mismatch")),
    )

    with pytest.raises(SpiceOperatorError, match="CUDA runtime initialization failed"):
        ensure_device_runtime_ready(
            requested_device="cuda",
            resolved_device=torch.device("cuda"),
        )


def test_ensure_device_runtime_ready_rejects_auto_cpu_when_cuda_devices_are_visible(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "11.8")

    with pytest.raises(
        SpiceOperatorError,
        match="CUDA devices are visible but the PyTorch CUDA runtime is unusable",
    ):
        ensure_device_runtime_ready(
            requested_device="auto",
            resolved_device=torch.device("cpu"),
        )


def test_recurrent_families_disable_auto_compile_on_cuda() -> None:
    training = SimpleNamespace(compile=CompileMode.AUTO)

    lstm_enabled = resolve_compile_enabled(
        training,
        device=torch.device("cuda"),
        precision="bf16-mixed",
        model_config=LstmModelConfig(
            input_projection_dim=8,
            hidden_size=16,
            num_layers=2,
            dropout=0.1,
            head_hidden_dim=8,
        ),
    )
    transformer_lstm_enabled = resolve_compile_enabled(
        training,
        device=torch.device("cuda"),
        precision="bf16-mixed",
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


def test_transformer_auto_compile_policy_is_unchanged_on_big_cuda_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=72),
    )
    training = SimpleNamespace(compile=CompileMode.AUTO)

    enabled = resolve_compile_enabled(
        training,
        device=torch.device("cuda"),
        precision="bf16-mixed",
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


def test_recurrent_families_default_to_fp32_on_cuda() -> None:
    assert (
        resolve_default_precision("lstm", torch.device("cuda"))
        is TrainingPrecision.FP32
    )
    assert (
        resolve_default_precision("transformer_lstm", torch.device("cuda"))
        is TrainingPrecision.FP32
    )


def test_transformer_default_precision_policy_is_unchanged_on_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert (
        resolve_default_precision("transformer", torch.device("cuda"))
        is TrainingPrecision.BF16_MIXED
    )

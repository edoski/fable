from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spice.config.models import TrainingConfig
from spice.modeling._runtime import CudaMemorySnapshot
from spice.modeling.evaluation_runtime import EvaluationScoringRuntimePlan
from spice.modeling.representations import RepresentationRuntimeContext
from spice.modeling.training_runtime import plan_training_runtime


def _training_config() -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "learning_rate": 0.02,
            "weight_decay": 0.03,
            "batch_size": 1,
            "max_epochs": 2,
            "early_stopping": {"patience": 2, "min_delta": 0.01},
            "gradient_clip_norm": 1.0,
            "seed": 7,
            "deterministic": True,
            "log_every_n_steps": 1,
        }
    )


def test_plan_training_runtime_uses_unshuffled_host_warmup_and_reuses_state(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    model.weight.data.fill_(4.0)
    original_weight = model.weight.detach().clone()
    runtime_context = RepresentationRuntimeContext(
        batch_size=1,
        available_host_memory_bytes=1024,
        available_device_memory_bytes=999,
    )
    calls = []
    prediction_state = object()

    def fake_build_prediction_batch_plan(*_args, runtime_context, seed, shuffle, **_kwargs):
        calls.append(
            {
                "budget": runtime_context.available_device_memory_bytes,
                "seed": seed,
                "shuffle": shuffle,
            }
        )
        return SimpleNamespace(source=[SimpleNamespace()])

    class FakeOptimizer:
        def __init__(self, params, *, lr, weight_decay) -> None:
            self.params = list(params)
            self.lr = lr
            self.weight_decay = weight_decay

    optimizers: list[FakeOptimizer] = []

    def fake_adamw(params, *, lr, weight_decay):
        optimizer = FakeOptimizer(params, lr=lr, weight_decay=weight_decay)
        optimizers.append(optimizer)
        return optimizer

    def fake_execute_training_batch(model, _batch, **_kwargs) -> object:
        model.weight.data.fill_(9.0)
        return object()

    empty_cache_calls = []
    fit_state_calls = []

    prediction_contract = SimpleNamespace(
        fit_training_state=lambda *args, **kwargs: fit_state_calls.append((args, kwargs))
        or prediction_state
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.build_prediction_batch_plan",
        fake_build_prediction_batch_plan,
    )
    monkeypatch.setattr("spice.modeling.training_runtime.torch.optim.AdamW", fake_adamw)
    monkeypatch.setattr(
        "spice.modeling.training_runtime.execute_training_batch",
        fake_execute_training_batch,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.snapshot_cuda_memory",
        lambda _device: CudaMemorySnapshot(
            free_bytes=1000,
            total_bytes=10000,
            allocated_bytes=10,
            reserved_bytes=100,
        ),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.reset_cuda_peak_memory",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.peak_cuda_reserved_bytes",
        lambda _device: 200,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.compute_device_resident_budget",
        lambda **kwargs: kwargs["free_bytes"]
        - (kwargs["peak_reserved_bytes"] - kwargs["baseline_reserved_bytes"]),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.torch.cuda.empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    plan = plan_training_runtime(
        model,
        prediction_contract=prediction_contract,
        execution_policy=SimpleNamespace(),
        representation_contract=SimpleNamespace(),
        store=SimpleNamespace(),
        train_sample_indices=np.array([0], dtype=np.int64),
        validation_sample_indices=np.array([1], dtype=np.int64),
        base_runtime_context=runtime_context,
        resolved_device=torch.device("cpu"),
        training_config=_training_config(),
        precision="32-true",
    )

    assert calls == [
        {"budget": 0, "seed": 7, "shuffle": False},
        {"budget": 900, "seed": 7, "shuffle": True},
        {"budget": 900, "seed": 7, "shuffle": False},
    ]
    assert optimizers[0].lr == 0.02
    assert optimizers[0].weight_decay == 0.03
    assert len(fit_state_calls) == 1
    assert plan.prediction_training_state is prediction_state
    assert plan.runtime_context.available_device_memory_bytes == 900
    assert plan.evaluation_scoring_runtime_plan == EvaluationScoringRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=plan.runtime_context,
        deterministic=None,
        seed=0,
    )
    assert torch.equal(model.weight.detach(), original_weight)
    assert empty_cache_calls == [True]


def test_plan_training_runtime_restores_model_and_clears_cache_after_probe_failure(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    model.weight.data.fill_(4.0)
    original_weight = model.weight.detach().clone()
    runtime_context = RepresentationRuntimeContext(
        batch_size=1,
        available_host_memory_bytes=1024,
        available_device_memory_bytes=999,
    )
    empty_cache_calls = []

    monkeypatch.setattr(
        "spice.modeling.training_runtime.build_prediction_batch_plan",
        lambda *_, **__: SimpleNamespace(source=[SimpleNamespace()]),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.torch.optim.AdamW",
        lambda params, **_kwargs: SimpleNamespace(params=list(params)),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.snapshot_cuda_memory",
        lambda _device: CudaMemorySnapshot(
            free_bytes=1000,
            total_bytes=10000,
            allocated_bytes=10,
            reserved_bytes=100,
        ),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.reset_cuda_peak_memory",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.torch.cuda.empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    def failing_execute_training_batch(model, *_args, **_kwargs) -> object:
        model.weight.data.fill_(9.0)
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        "spice.modeling.training_runtime.execute_training_batch",
        failing_execute_training_batch,
    )
    prediction_contract = SimpleNamespace(
        fit_training_state=lambda *_args, **_kwargs: object()
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        plan_training_runtime(
            model,
            prediction_contract=prediction_contract,
            execution_policy=SimpleNamespace(),
            representation_contract=SimpleNamespace(),
            store=SimpleNamespace(),
            train_sample_indices=np.array([0], dtype=np.int64),
            validation_sample_indices=np.array([1], dtype=np.int64),
            base_runtime_context=runtime_context,
            resolved_device=torch.device("cpu"),
            training_config=_training_config(),
            precision="32-true",
        )

    assert torch.equal(model.weight.detach(), original_weight)
    assert empty_cache_calls == [True]

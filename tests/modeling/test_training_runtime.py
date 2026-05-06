from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from spice.config.models import TrainingConfig
from spice.modeling.dataset_builders import PreparedTrainingSampleSelection
from spice.modeling.representations import DeviceStorageBudget, RepresentationRuntimeContext
from spice.modeling.runtime_planning import ModelingRuntimePlan
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


def _runtime_plan(runtime_context: RepresentationRuntimeContext) -> ModelingRuntimePlan:
    return ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=runtime_context,
        deterministic=True,
        seed=7,
        compile_enabled=True,
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
        device_storage_budget=DeviceStorageBudget.coarse(999),
    )
    calls = []
    prediction_state = object()
    train_facts = object()
    validation_facts = object()
    execution_policy = SimpleNamespace()

    def fake_build_prediction_batch_plan(
        *_args, temporal_facts, runtime_context, seed, shuffle, **_kwargs
    ):
        calls.append(
            {
                "budget": runtime_context.device_storage_budget,
                "host_loader_policy": runtime_context.host_loader_policy,
                "seed": seed,
                "shuffle": shuffle,
                "temporal_facts": temporal_facts,
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
        fit_training_state=lambda *args, **kwargs: (
            fit_state_calls.append((args, kwargs)) or prediction_state
        )
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
        "spice.modeling.training_runtime.measure_device_resident_budget",
        lambda *, run_probe, **_kwargs: run_probe() or 900,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.torch.cuda.empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    plan = plan_training_runtime(
        cast(Any, model),
        prediction_contract=cast(Any, prediction_contract),
        execution_policy=cast(Any, execution_policy),
        representation_contract=cast(Any, SimpleNamespace()),
        store=cast(Any, SimpleNamespace()),
        train_samples=PreparedTrainingSampleSelection("train", cast(Any, train_facts)),
        validation_samples=PreparedTrainingSampleSelection(
            "validation",
            cast(Any, validation_facts),
        ),
        runtime_plan=_runtime_plan(runtime_context),
        training_config=_training_config(),
    )

    assert calls == [
        {
            "budget": DeviceStorageBudget.disabled(),
            "host_loader_policy": "single_process_unpinned",
            "seed": 7,
            "shuffle": False,
            "temporal_facts": train_facts,
        },
        {
            "budget": DeviceStorageBudget.measured(900),
            "host_loader_policy": "automatic",
            "seed": 7,
            "shuffle": True,
            "temporal_facts": train_facts,
        },
        {
            "budget": DeviceStorageBudget.measured(900),
            "host_loader_policy": "automatic",
            "seed": 7,
            "shuffle": False,
            "temporal_facts": validation_facts,
        },
    ]
    assert optimizers[0].lr == 0.02
    assert optimizers[0].weight_decay == 0.03
    assert len(fit_state_calls) == 1
    assert fit_state_calls[0][1]["temporal_facts"] is train_facts
    assert plan.prediction_training_state is prediction_state
    assert plan.runtime_plan == ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        representation_runtime_context=runtime_context.with_device_storage_budget(
            DeviceStorageBudget.measured(900)
        ),
        deterministic=True,
        seed=7,
        compile_enabled=True,
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
        device_storage_budget=DeviceStorageBudget.coarse(999),
    )
    empty_cache_calls = []
    execution_policy = SimpleNamespace()

    monkeypatch.setattr(
        "spice.modeling.training_runtime.build_prediction_batch_plan",
        lambda *_, **__: SimpleNamespace(source=[SimpleNamespace()]),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.torch.optim.AdamW",
        lambda params, **_kwargs: SimpleNamespace(params=list(params)),
    )
    monkeypatch.setattr(
        "spice.modeling.training_runtime.measure_device_resident_budget",
        lambda *, run_probe, **_kwargs: run_probe(),
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
    prediction_contract = SimpleNamespace(fit_training_state=lambda *_args, **_kwargs: object())

    with pytest.raises(RuntimeError, match="probe failed"):
        plan_training_runtime(
            cast(Any, model),
            prediction_contract=cast(Any, prediction_contract),
            execution_policy=cast(Any, execution_policy),
            representation_contract=cast(Any, SimpleNamespace()),
            store=cast(Any, SimpleNamespace()),
            train_samples=PreparedTrainingSampleSelection(
                "train",
                cast(Any, object()),
            ),
            validation_samples=PreparedTrainingSampleSelection(
                "validation",
                cast(Any, object()),
            ),
            runtime_plan=_runtime_plan(runtime_context),
            training_config=_training_config(),
        )

    assert torch.equal(model.weight.detach(), original_weight)
    assert empty_cache_calls == [True]

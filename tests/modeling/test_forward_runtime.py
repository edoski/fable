from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spice.modeling.forward_runtime import (
    run_planned_model_input_forward,
    run_planned_prediction_forward,
)
from spice.modeling.representations import DeviceStorageBudget, RepresentationRuntimeContext


def test_planned_model_input_forward_uses_host_warmup_then_measured_budget(
    monkeypatch,
) -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=2,
        available_host_memory_bytes=1024,
        device_storage_budget=DeviceStorageBudget.coarse(999),
    )
    sources = [[SimpleNamespace(name="warmup")], [SimpleNamespace(name="final")]]
    budgets: list[DeviceStorageBudget] = []
    execution_policy = SimpleNamespace(name="policy")
    policies = []

    def fake_build_model_input_batch_plan(*_args, runtime_context, execution_policy, **_kwargs):
        budgets.append(runtime_context.device_storage_budget)
        policies.append(execution_policy)
        return SimpleNamespace(source=sources[len(budgets) - 1])

    measured_sources = []
    seen_batches = []

    def fake_forward(_model, *, loader, on_outputs, **_kwargs):
        for batch in loader:
            on_outputs(batch, SimpleNamespace())

    monkeypatch.setattr(
        "spice.modeling.forward_runtime.build_model_input_batch_plan",
        fake_build_model_input_batch_plan,
    )
    monkeypatch.setattr(
        "spice.modeling.forward_runtime._measure_forward_batch_budget",
        lambda _model, *, loader, **_kwargs: measured_sources.append(loader) or 77,
    )
    monkeypatch.setattr(
        "spice.modeling.forward_runtime.run_model_forward_pass",
        fake_forward,
    )

    run_planned_model_input_forward(
        SimpleNamespace(),
        store=SimpleNamespace(),
        sample_indices=np.array([0], dtype=np.int64),
        representation_contract=SimpleNamespace(),
        execution_policy=execution_policy,
        base_runtime_context=runtime_context,
        resolved_device=torch.device("cpu"),
        precision="32-true",
        seed=3,
        on_outputs=lambda batch, _outputs: seen_batches.append(batch),
    )

    assert budgets == [DeviceStorageBudget.disabled(), DeviceStorageBudget.measured(77)]
    assert policies == [execution_policy, execution_policy]
    assert measured_sources == [sources[0]]
    assert seen_batches == sources[1]


def test_planned_prediction_forward_preserves_callback_and_final_source(
    monkeypatch,
) -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=2,
        available_host_memory_bytes=1024,
        device_storage_budget=DeviceStorageBudget.coarse(999),
    )
    sources = [[SimpleNamespace(name="warmup")], [SimpleNamespace(name="final")]]
    budgets: list[DeviceStorageBudget] = []
    shuffles: list[bool | None] = []

    def fake_build_prediction_batch_plan(*_args, runtime_context, shuffle=None, **_kwargs):
        budgets.append(runtime_context.device_storage_budget)
        shuffles.append(shuffle)
        return SimpleNamespace(source=sources[len(budgets) - 1])

    seen_batches = []

    def fake_forward(_model, *, loader, on_outputs, **_kwargs):
        for batch in loader:
            on_outputs(batch, SimpleNamespace())

    monkeypatch.setattr(
        "spice.modeling.forward_runtime.build_prediction_batch_plan",
        fake_build_prediction_batch_plan,
    )
    monkeypatch.setattr(
        "spice.modeling.forward_runtime._measure_forward_batch_budget",
        lambda _model, *, loader, **_kwargs: 55 if loader == sources[0] else 0,
    )
    monkeypatch.setattr(
        "spice.modeling.forward_runtime.run_model_forward_pass",
        fake_forward,
    )

    run_planned_prediction_forward(
        SimpleNamespace(),
        store=SimpleNamespace(),
        sample_indices=np.array([0], dtype=np.int64),
        representation_contract=SimpleNamespace(),
        prediction_contract=SimpleNamespace(),
        execution_policy=SimpleNamespace(),
        base_runtime_context=runtime_context,
        resolved_device=torch.device("cpu"),
        precision="32-true",
        seed=3,
        on_outputs=lambda batch, _outputs: seen_batches.append(batch),
    )

    assert budgets == [DeviceStorageBudget.disabled(), DeviceStorageBudget.measured(55)]
    assert shuffles == [False, False]
    assert seen_batches == sources[1]


def test_planned_forward_rejects_empty_sample_indices() -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=2,
        available_host_memory_bytes=1024,
    )

    with pytest.raises(ValueError, match="sample_indices must be non-empty"):
        run_planned_model_input_forward(
            SimpleNamespace(),
            store=SimpleNamespace(),
            sample_indices=np.array([], dtype=np.int64),
            representation_contract=SimpleNamespace(),
            execution_policy=SimpleNamespace(),
            base_runtime_context=runtime_context,
            resolved_device=torch.device("cpu"),
            precision="32-true",
            seed=3,
            on_outputs=lambda _batch, _outputs: None,
        )

    with pytest.raises(ValueError, match="sample_indices must be non-empty"):
        run_planned_prediction_forward(
            SimpleNamespace(),
            store=SimpleNamespace(),
            sample_indices=np.array([], dtype=np.int64),
            representation_contract=SimpleNamespace(),
            prediction_contract=SimpleNamespace(),
            execution_policy=SimpleNamespace(),
            base_runtime_context=runtime_context,
            resolved_device=torch.device("cpu"),
            precision="32-true",
            seed=3,
            on_outputs=lambda _batch, _outputs: None,
        )

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from spice.config.models import TrainingConfig
from spice.metrics import MetricSet
from spice.modeling.objective_runtime import CompiledObjectiveRuntime
from spice.modeling.representations import RepresentationRuntimeContext
from spice.modeling.runtime_planning import ModelingRuntimePlan
from spice.modeling.training_runner import (
    TrainingCallbacks,
    TrainingFitSpec,
    TrainingMetricEvaluationSpec,
    evaluate_training_metrics,
    run_training_fit,
)
from spice.modeling.training_runtime import PreparedTrainingRuntime, TrainingRuntimePlan
from spice.objectives import CompiledObjectiveContract


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


def _training_config(*, max_epochs: int = 2, patience: int = 2) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 1,
            "max_epochs": max_epochs,
            "early_stopping": {"patience": patience, "min_delta": 0.01},
            "gradient_clip_norm": 1.0,
            "seed": 1,
            "deterministic": True,
            "log_every_n_steps": 1,
        }
    )


def _objective_contract() -> CompiledObjectiveContract:
    return CompiledObjectiveContract(
        objective_id="validation",
        metric_id="score",
        direction="maximize",
        benchmark_id=None,
    )


def _objective_runtime(evaluate_metrics_fn=None) -> CompiledObjectiveRuntime:
    return CompiledObjectiveRuntime(
        contract=_objective_contract(),
        evaluate_metrics_fn=evaluate_metrics_fn
        or (lambda validation_metrics, scoring_input_factory: validation_metrics),
    )


def _patch_training_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=1,
        available_host_memory_bytes=1024,
    )
    runtime_plan = ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="fp32",
        representation_runtime_context=runtime_context,
        deterministic=None,
        seed=0,
    )

    def fake_prepare_training_runtime(model, **_kwargs):
        return PreparedTrainingRuntime(
            runtime_plan=runtime_plan,
            fit_model=model,
            optimizer=cast(torch.optim.Optimizer, SimpleNamespace()),
            batch_plan=TrainingRuntimePlan(
                runtime_context=runtime_context,
                train_batch_plan=cast(Any, SimpleNamespace(source=[0])),
                validation_batch_plan=cast(Any, SimpleNamespace(source=[0])),
                evaluation_runtime_plan=runtime_plan,
                prediction_training_state=None,
            ),
        )

    monkeypatch.setattr(
        "spice.modeling.training_runner.prepare_training_runtime",
        fake_prepare_training_runtime,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runner.modeling_backend_scope",
        lambda _plan: nullcontext(),
    )


def _fit_spec(
    tmp_path,
    *,
    model,
    training_config,
    objective_runtime=None,
) -> TrainingFitSpec:
    return TrainingFitSpec(
        model=model,
        model_config=cast(Any, SimpleNamespace()),
        prediction_contract=cast(Any, SimpleNamespace(fit_training_state=lambda *_, **__: None)),
        representation_contract=cast(Any, SimpleNamespace()),
        objective_runtime=objective_runtime or _objective_runtime(),
        execution_policy=cast(Any, SimpleNamespace()),
        store=cast(Any, SimpleNamespace()),
        train_sample_indices=np.array([0], dtype=np.int64),
        validation_sample_indices=np.array([0], dtype=np.int64),
        training_config=training_config,
    )


def test_training_fit_restores_best_state_and_calls_early_stop_callback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch_state = {"epoch": 0}
    early_stop_calls: list[tuple[int, int]] = []
    objective_values = [1.0, 1.005]

    def fake_run_epoch(model, *, training, **_kwargs):
        if training:
            epoch_state["epoch"] += 1
            model.weight.data.fill_(float(epoch_state["epoch"]))
        return MetricSet({"score": objective_values[epoch_state["epoch"] - 1]})

    _patch_training_runtime(monkeypatch)
    monkeypatch.setattr("spice.modeling.training_runner.run_epoch", fake_run_epoch)
    monkeypatch.setattr(
        "spice.modeling.training_runner.ModelScoringInput",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("validation objective must not build scoring input")
        ),
    )

    model = _TinyModel()
    result = run_training_fit(
        _fit_spec(
            tmp_path,
            model=model,
            training_config=_training_config(max_epochs=2, patience=1),
        ),
        callbacks=TrainingCallbacks(
            on_early_stop=lambda epoch, best_epoch: early_stop_calls.append(
                (epoch, best_epoch)
            ),
        ),
    )

    assert result.best_epoch == 1
    assert result.best_objective_value == 1.0
    assert len(result.train_history) == 2
    assert len(result.validation_history) == 2
    assert len(result.objective_history) == 2
    assert model.weight.item() == 1.0
    assert early_stop_calls == [(2, 1)]


def test_evaluate_training_metrics_uses_batch_plan_and_prediction_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_context = RepresentationRuntimeContext(
        batch_size=1,
        available_host_memory_bytes=1024,
    )
    runtime_plan = ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="fp32",
        representation_runtime_context=runtime_context,
        deterministic=True,
        seed=1,
    )
    monkeypatch.setattr(
        "spice.modeling.training_runner.build_cuda_modeling_runtime_plan",
        lambda **_: runtime_plan,
    )
    monkeypatch.setattr("spice.modeling.training_runner.modeling_backend_scope", nullcontext)

    seen_training_states: list[object | None] = []

    class Accumulator:
        def __init__(self) -> None:
            self.values: list[float] = []

        def update(self, batch_state) -> None:
            self.values.append(batch_state["score"])

        def finalize(self) -> MetricSet:
            return MetricSet({"score": sum(self.values)})

    prediction_state = object()
    prediction_contract = SimpleNamespace(
        create_epoch_accumulator=Accumulator,
        compute_batch_loss_and_state=lambda outputs, targets, training_state: (
            seen_training_states.append(training_state) or torch.tensor(0.0),
            {"score": 2.5},
        ),
    )

    forward_calls = []

    def fake_run_planned_prediction_forward(_model, *, on_outputs, **kwargs):
        forward_calls.append(kwargs)
        on_outputs(SimpleNamespace(targets="metrics"), outputs=SimpleNamespace())

    monkeypatch.setattr(
        "spice.modeling.training_runner.run_planned_prediction_forward",
        fake_run_planned_prediction_forward,
    )

    metrics = evaluate_training_metrics(
        TrainingMetricEvaluationSpec(
            model=_TinyModel(),
            model_config=cast(Any, SimpleNamespace()),
            prediction_contract=cast(Any, prediction_contract),
            execution_policy=cast(Any, SimpleNamespace()),
            representation_contract=cast(Any, SimpleNamespace()),
            store=cast(Any, SimpleNamespace()),
            sample_indices=np.array([0], dtype=np.int64),
            prediction_training_state=prediction_state,
            training_config=_training_config(),
        )
    )

    assert forward_calls[0]["base_runtime_context"] is runtime_context
    assert forward_calls[0]["seed"] == 1
    assert seen_training_states == [prediction_state]
    assert metrics.require("score") == 2.5

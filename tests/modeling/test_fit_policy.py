from __future__ import annotations

import pytest
import torch

from spice.core.errors import SpiceOperatorError
from spice.metrics import MetricSet
from spice.modeling._fit_policy import CompletedEpoch, TrainingFitPolicy
from spice.objectives import CompiledObjectiveContract


def _objective(*, direction: str = "maximize") -> CompiledObjectiveContract:
    return CompiledObjectiveContract(
        objective_id="validation",
        metric_id="score",
        direction=direction,
        benchmark_id=None,
    )


def _policy(*, direction: str = "maximize", patience: int = 2) -> TrainingFitPolicy:
    return TrainingFitPolicy.create(
        objective_contract=_objective(direction=direction),
        max_epochs=3,
        patience=patience,
        min_delta=0.1,
    )


def _completed(epoch: int, score: float) -> CompletedEpoch:
    metrics = MetricSet({"score": score})
    return CompletedEpoch(
        epoch=epoch,
        train_metrics=metrics,
        validation_metrics=metrics,
        objective_metrics=metrics,
    )


def test_fit_policy_appends_histories_and_uses_strict_min_delta() -> None:
    model = torch.nn.Linear(1, 1)
    policy = _policy()

    first = policy.record_completed_epoch(_completed(1, 1.0), model=model)
    second = policy.record_completed_epoch(_completed(2, 1.05), model=model)

    assert len(policy.objective_history) == 2
    assert len(policy.train_history) == 2
    assert len(policy.validation_history) == 2
    assert first.progress is not None
    assert first.progress.best_epoch == 1
    assert second.progress is not None
    assert second.progress.best_epoch == 1
    assert second.should_stop is False


def test_fit_policy_nonfinite_before_best_state_raises_without_append() -> None:
    policy = _policy()

    with pytest.raises(SpiceOperatorError, match="before any valid best state"):
        policy.handle_nonfinite_metrics(
            epoch=1,
            phase="train",
            metrics=MetricSet({"score": float("nan")}),
        )

    assert policy.objective_history == []


def test_fit_policy_nonfinite_after_best_state_stops_without_progress() -> None:
    model = torch.nn.Linear(1, 1)
    policy = _policy()
    policy.record_completed_epoch(_completed(1, 1.0), model=model)

    decision = policy.handle_nonfinite_metrics(
        epoch=2,
        phase="validation",
        metrics=MetricSet({"score": float("nan")}),
    )

    assert decision is not None
    assert decision.should_stop is True
    assert decision.progress is None
    assert decision.early_stop == (2, 1)
    assert len(policy.objective_history) == 1


def test_fit_policy_patience_stop_and_minimize_direction() -> None:
    model = torch.nn.Linear(1, 1)
    policy = _policy(direction="minimize", patience=1)

    first = policy.record_completed_epoch(_completed(1, 1.0), model=model)
    second = policy.record_completed_epoch(_completed(2, 1.05), model=model)

    assert first.should_stop is False
    assert second.should_stop is True
    assert second.early_stop == (2, 1)
    assert second.progress is not None
    assert second.progress.direction == "minimize"
    assert second.progress.best_objective_value == 1.0

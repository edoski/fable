from __future__ import annotations

from types import SimpleNamespace

import torch

from spice.metrics import MetricSet
from spice.modeling._epoch_execution import execute_training_batch, run_epoch


class _Batch:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.targets = object()

    def to_device(self, _device: torch.device):
        self.events.append("to_device")
        return self

    def model_kwargs(self):
        return {"x": torch.tensor([1.0])}


class _Model(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.events.append(f"forward:{self.training}:{torch.is_grad_enabled()}")
        return x * self.weight


class _Optimizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True
        self.events.append("zero_grad")

    def step(self) -> None:
        self.events.append("step")


def test_execute_training_batch_preserves_backward_clip_step_order(monkeypatch) -> None:
    events: list[str] = []
    model = _Model(events)
    batch = _Batch(events)
    optimizer = _Optimizer(events)
    prediction_contract = SimpleNamespace(
        compute_batch_loss_and_state=lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"count": 1},
        )
    )
    monkeypatch.setattr(
        "spice.modeling._epoch_execution.torch.nn.utils.clip_grad_norm_",
        lambda _params, _norm: events.append("clip"),
    )

    state = execute_training_batch(
        model,
        batch,
        resolved_device=torch.device("cpu"),
        precision="32-true",
        prediction_contract=prediction_contract,
        prediction_training_state=object(),
        optimizer=optimizer,
        gradient_clip_norm=1.0,
    )

    assert state == {"count": 1}
    assert events == [
        "to_device",
        "zero_grad",
        "forward:True:True",
        "clip",
        "step",
    ]


def test_run_epoch_validation_uses_eval_mode_and_disabled_grad() -> None:
    events: list[str] = []
    model = _Model(events)

    class Accumulator:
        def __init__(self) -> None:
            self.states = []

        def update(self, state) -> None:
            self.states.append(state)

        def finalize(self) -> MetricSet:
            return MetricSet({"count": float(len(self.states))})

    prediction_contract = SimpleNamespace(
        create_epoch_accumulator=Accumulator,
        compute_batch_loss_and_state=lambda *_args, **_kwargs: (
            torch.tensor(0.0),
            {"count": 1},
        ),
    )

    metrics = run_epoch(
        model,
        loader=[_Batch(events)],
        resolved_device=torch.device("cpu"),
        precision="32-true",
        prediction_contract=prediction_contract,
        prediction_training_state=None,
        optimizer=None,
        gradient_clip_norm=None,
        training=False,
    )

    assert events == ["to_device", "forward:False:False"]
    assert metrics.require("count") == 1.0

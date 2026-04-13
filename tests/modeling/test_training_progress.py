from __future__ import annotations

from types import SimpleNamespace

import torch

from spice.config import coerce_prediction_config
from spice.core.reporting import NullReporter
from spice.modeling.training import ReporterProgressCallback
from spice.prediction import compile_prediction_contract


class CaptureReporter(NullReporter):
    def __init__(self) -> None:
        self.messages: list[str | None] = []

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> int:
        del name, total, unit
        return 1

    def update_task(
        self,
        task_id: int,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        del task_id, completed, advance
        self.messages.append(message)


def test_reporter_progress_callback_smooths_training_loss() -> None:
    reporter = CaptureReporter()
    prediction = coerce_prediction_config(
        {
            "id": "candidate_slate_current",
            "family": {"id": "candidate_slate_current"},
        }
    )
    callback = ReporterProgressCallback(
        reporter,
        max_epochs=5,
        prediction_contract=compile_prediction_contract(
            prediction_id=prediction.id,
            family_config=prediction.family,
        ),
    )
    trainer = SimpleNamespace(num_training_batches=10, current_epoch=0)

    callback.on_train_start(trainer, SimpleNamespace())
    callback.on_train_batch_end(
        trainer,
        SimpleNamespace(),
        {"loss": torch.tensor(10.0)},
        None,
        0,
    )
    callback.on_train_batch_end(
        trainer,
        SimpleNamespace(),
        {"loss": torch.tensor(0.0)},
        None,
        1,
    )

    assert reporter.messages[0] == "epoch=1/5 batch 1/10 loss=10.0"
    assert reporter.messages[1] == "epoch=1/5 batch 2/10 loss=8.80"

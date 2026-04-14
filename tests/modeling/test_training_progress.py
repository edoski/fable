from __future__ import annotations

from types import SimpleNamespace

import torch

from spice.config import coerce_prediction_config
from spice.core.reporting import NullReporter, StageMetricValue
from spice.modeling.training import ReporterProgressCallback
from spice.prediction import compile_prediction_contract


class CaptureReporter(NullReporter):
    def __init__(self) -> None:
        self.messages: list[str | None] = []
        self.metrics: list[dict[str, str]] = []

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
        metrics: tuple[StageMetricValue, ...] = (),
    ) -> None:
        del task_id, completed, advance
        self.messages.append(message)
        self.metrics.append({metric.id: metric.value for metric in metrics})


def test_reporter_progress_callback_smooths_training_loss() -> None:
    reporter = CaptureReporter()
    prediction = coerce_prediction_config(
        {
            "id": "candidate_offset_selection",
            "family": {"id": "candidate_offset_selection"},
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

    assert reporter.messages[0] == "batch 1/10"
    assert reporter.messages[1] == "batch 2/10"
    assert reporter.metrics[0] == {"epoch": "1/5", "total_loss": "10.0"}
    assert reporter.metrics[1] == {"epoch": "1/5", "total_loss": "8.80"}

from __future__ import annotations

from types import SimpleNamespace

import torch

from spice.config import TrainingConfig, coerce_prediction_config
from spice.core.reporting import NullReporter, StageMetricValue
from spice.modeling.lightning_module import TemporalLightningModule
from spice.modeling.models import ModelOutputs, TemporalModel
from spice.modeling.training import ReporterProgressCallback
from spice.prediction import MetricSet, compile_prediction_contract


class DummyTemporalModel(TemporalModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **model_kwargs: torch.Tensor) -> ModelOutputs:
        del model_kwargs
        raise NotImplementedError


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
        completed: int | None = None,
    ) -> int:
        del name, total, unit, completed
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


def test_reporter_progress_callback_reports_accumulator_snapshot_with_throttling() -> None:
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
        log_every_n_steps=2,
        prediction_contract=compile_prediction_contract(
            prediction_id=prediction.id,
            family_config=prediction.family,
        ),
    )
    trainer = SimpleNamespace(num_training_batches=10, current_epoch=0)
    pl_module = TemporalLightningModule(
        DummyTemporalModel(),
        training_config=TrainingConfig.model_validate(
            {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 8,
                "max_epochs": 5,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
                "gradient_clip_norm": 1.0,
                "device": "cpu",
                "seed": 2026,
                "deterministic": True,
                "log_every_n_steps": 2,
                "precision": "fp32",
                "compile": "off",
            }
        ),
        prediction_contract=compile_prediction_contract(
            prediction_id=prediction.id,
            family_config=prediction.family,
        ),
        prediction_training_state=None,
    )
    pl_module.train_progress_snapshot = lambda: MetricSet(
            values={
                "total_loss": 8.8,
                "exact_optimum_hit_rate": 0.4,
                "cost_over_optimum": 0.2,
                "profit_over_baseline": 0.1,
            }
        )

    callback.on_train_start(trainer, pl_module)
    callback.on_train_batch_end(
        trainer,
        pl_module,
        {"loss": torch.tensor(10.0)},
        None,
        0,
    )
    callback.on_train_batch_end(
        trainer,
        pl_module,
        {"loss": torch.tensor(0.0)},
        None,
        1,
    )

    assert reporter.messages == ["batch 2/10"]
    assert reporter.metrics == [
        {
            "epoch": "1/5",
            "profit_over_baseline": "0.100",
            "cost_over_optimum": "0.200",
            "total_loss": "8.80",
            "exact_optimum_hit_rate": "0.400",
        }
    ]

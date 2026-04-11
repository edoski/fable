"""Lightning module for temporal SPICE models."""

from __future__ import annotations

import lightning as L
import torch

from ..core.config import TrainingConfig
from .evaluation import BatchMetrics, EpochMetrics, compute_temporal_batch_metrics, mean_metrics
from .models import ModelOutputs, TemporalModel


class TemporalLightningModule(L.LightningModule):
    """Training harness that keeps the project-specific task semantics custom."""

    def __init__(
        self,
        model: TemporalModel,
        *,
        class_weights: torch.Tensor,
        action_count: int,
        training_config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("class_weights", class_weights, persistent=False)
        self.action_count = action_count
        self.training_config = training_config
        self.train_history: list[EpochMetrics] = []
        self.validation_history: list[EpochMetrics] = []
        self._train_batches: list[BatchMetrics] = []
        self._validation_batches: list[BatchMetrics] = []

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        return self.model(inputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def on_train_epoch_start(self) -> None:
        self._train_batches = []

    def on_validation_epoch_start(self) -> None:
        self._validation_batches = []

    def _shared_step(self, batch: dict[str, torch.Tensor], *, stage: str) -> torch.Tensor:
        outputs = self.model(batch["inputs"])
        total_loss, metrics = compute_temporal_batch_metrics(
            outputs,
            batch,
            class_weights=self.get_buffer("class_weights"),
            training_config=self.training_config,
        )
        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
        )
        self.log(
            f"{stage}/accuracy",
            metrics.correct_count / metrics.count,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
        )
        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_accuracy",
            metrics.correct_count / metrics.count,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if stage == "train":
            self._train_batches.append(metrics)
        else:
            self._validation_batches.append(metrics)
        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="validation")

    def on_train_epoch_end(self) -> None:
        if self._train_batches:
            metrics = mean_metrics(self._train_batches)
            self.train_history.append(metrics)
            self.log(
                "train/profit_over_baseline",
                metrics.mean_profit_over_baseline,
                prog_bar=False,
            )
            self.log("train/cost_over_optimum", metrics.mean_cost_over_optimum, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        if self._validation_batches:
            metrics = mean_metrics(self._validation_batches)
            self.validation_history.append(metrics)
            self.log(
                "validation/profit_over_baseline",
                metrics.mean_profit_over_baseline,
                prog_bar=True,
            )
            self.log(
                "validation/cost_over_optimum",
                metrics.mean_cost_over_optimum,
                prog_bar=False,
            )

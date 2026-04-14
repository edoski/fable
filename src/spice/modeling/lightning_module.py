"""Lightning module for temporal SPICE models."""

from __future__ import annotations

import lightning as L
import torch

from ..config import TrainingConfig
from ..prediction import CompiledPredictionContract, MetricSet
from .models import ModelOutputs, TemporalModel


class TemporalLightningModule(L.LightningModule):
    """Training harness for temporal SPICE models."""

    def __init__(
        self,
        model: TemporalModel,
        *,
        training_config: TrainingConfig,
        prediction_contract: CompiledPredictionContract,
        prediction_training_state: object | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.prediction_contract = prediction_contract
        self.prediction_training_state = prediction_training_state
        self.train_history: list[MetricSet] = []
        self.validation_history: list[MetricSet] = []
        self._train_batches: list[object] = []
        self._validation_batches: list[object] = []

    def forward(self, **model_kwargs: torch.Tensor) -> ModelOutputs:
        return self.model(**model_kwargs)

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

    def _log_metric_set(self, stage: str, metrics: MetricSet) -> None:
        for descriptor in self.prediction_contract.training_metric_descriptors:
            if descriptor.id not in metrics.values:
                continue
            value = metrics.values[descriptor.id]
            self.log(
                f"{stage}/{descriptor.id}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage != "train" and descriptor.role == "primary"),
            )
            self.log(
                f"{stage}_{descriptor.id}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def _shared_step(self, batch, *, stage: str) -> torch.Tensor:
        outputs = self.model(**batch.model_kwargs())
        loss, batch_state = self.prediction_contract.compute_batch_loss_and_state(
            outputs,
            batch.targets,
            training_state=self.prediction_training_state,
        )
        metric_set = self.prediction_contract.summarize_epoch_metrics([batch_state])
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._log_metric_set(stage, metric_set)

        if stage == "train":
            self._train_batches.append(batch_state)
        else:
            self._validation_batches.append(batch_state)
        return loss

    def training_step(self, batch, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="validation")

    def on_train_epoch_end(self) -> None:
        if self._train_batches:
            metrics = self.prediction_contract.summarize_epoch_metrics(self._train_batches)
            self.train_history.append(metrics)
            self._log_metric_set("train", metrics)

    def on_validation_epoch_end(self) -> None:
        if self._validation_batches:
            metrics = self.prediction_contract.summarize_epoch_metrics(self._validation_batches)
            self.validation_history.append(metrics)
            self._log_metric_set("validation", metrics)

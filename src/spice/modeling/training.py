"""Training utilities backed by Lightning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy.typing import NDArray

from ..config import ModelConfig, TrainingConfig
from ..core.console import NullReporter, Reporter
from ..data.datasets import TemporalDatasetStore
from ._runtime import (
    build_sequence_loader,
    resolve_compile_enabled,
    resolve_device,
    resolve_trainer_precision,
    set_global_seed,
)
from .datamodule import TemporalDataModule
from .evaluation import EpochMetrics, compute_temporal_batch_metrics, mean_metrics
from .lightning_module import TemporalLightningModule
from .models import TemporalModel
from .torch_datasets import build_class_weights, move_batch_to_device

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    train_history: list[EpochMetrics]
    validation_history: list[EpochMetrics]
    best_checkpoint_path: Path | None
    resolved_precision: str
    compiled: bool
    resolved_device: str


def _unwrap_compiled_model(model: TemporalModel) -> TemporalModel:
    return cast(TemporalModel, getattr(model, "_orig_mod", model))


class ReporterProgressCallback(L.Callback):
    def __init__(self, reporter: Reporter, *, max_epochs: int) -> None:
        super().__init__()
        self._reporter = reporter
        self._max_epochs = max_epochs
        self._task_id: int | None = None
        self._total_batches = 0
        self._train_batches_per_epoch = 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        train_batches = trainer.num_training_batches
        if not isinstance(train_batches, int) or train_batches <= 0:
            self._task_id = self._reporter.start_task("train epochs")
            return
        self._train_batches_per_epoch = train_batches
        self._total_batches = train_batches * self._max_epochs
        self._task_id = self._reporter.start_task(
            "train epochs",
            total=self._total_batches,
            unit="batches",
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del batch
        if self._task_id is None:
            return
        loss_value = _loss_value(outputs)
        message = (
            f"epoch={trainer.current_epoch + 1}/{self._max_epochs} "
            f"train={batch_idx + 1}/{max(1, self._train_batches_per_epoch)}"
        )
        if loss_value is not None:
            message = f"{message} loss={loss_value:.4f}"
        self._reporter.update_task(self._task_id, advance=1, message=message)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._task_id is None:
            return
        metrics = getattr(pl_module, "validation_history", None)
        if not metrics:
            return
        latest = metrics[-1]
        completed = None
        if self._train_batches_per_epoch > 0:
            completed = min(
                self._total_batches,
                (trainer.current_epoch + 1) * self._train_batches_per_epoch,
            )
        self._reporter.update_task(
            self._task_id,
            completed=completed,
            message=(
                f"epoch={trainer.current_epoch + 1}/{self._max_epochs} "
                f"validation_loss={latest.total_loss:.4f} "
                f"validation_accuracy={latest.accuracy:.3f}"
            ),
        )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer
        if self._task_id is None:
            return
        validation_history = getattr(pl_module, "validation_history", [])
        self._reporter.finish_task(
            self._task_id,
            message=f"best_epoch={_best_epoch(validation_history)}",
        )
        self._task_id = None


def _loss_value(outputs: object) -> float | None:
    if isinstance(outputs, torch.Tensor):
        return float(outputs.detach().item())
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
    return None


def _trainer_device_args(device_name: str) -> tuple[str, int | str | list[int]]:
    resolved = resolve_device(device_name)
    if resolved.type == "cuda":
        if resolved.index is None:
            return "gpu", 1
        return "gpu", [resolved.index]
    if resolved.type == "mps":
        return "mps", 1
    return "cpu", 1


def _best_epoch(validation_history: list[EpochMetrics]) -> int:
    if not validation_history:
        return 1
    return min(
        range(len(validation_history)),
        key=lambda index: validation_history[index].total_loss,
    ) + 1


def train_model(
    model: TemporalModel,
    *,
    model_config: ModelConfig,
    store: TemporalDatasetStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    lookback_steps: int,
    training_config: TrainingConfig,
    artifact_dir: Path,
    reporter: Reporter | None = None,
) -> TrainingResult:
    reporter = reporter or NullReporter()
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    set_global_seed(training_config.seed)
    L.seed_everything(training_config.seed, workers=True)
    device = resolve_device(training_config.device)
    precision = resolve_trainer_precision(
        training_config,
        device=device,
        model_config=model_config,
    )
    compile_enabled = resolve_compile_enabled(
        training_config,
        device=device,
        precision=precision,
        model_config=model_config,
    )
    fit_model = cast(TemporalModel, torch.compile(model) if compile_enabled else model)

    data_module = TemporalDataModule(
        store=store,
        train_sample_indices=train_sample_indices,
        validation_sample_indices=validation_sample_indices,
        lookback_steps=lookback_steps,
        batch_size=training_config.batch_size,
        device=device,
    )

    module = TemporalLightningModule(
        fit_model,
        class_weights=data_module.class_weights,
        action_count=store.action_count,
        training_config=training_config,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=artifact_dir / "checkpoints",
        filename="epoch={epoch:02d}-validation_loss={validation_loss:.6f}",
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=training_config.early_stopping.patience,
        min_delta=training_config.early_stopping.min_delta,
    )
    accelerator, devices = _trainer_device_args(training_config.device)
    callbacks: list[L.Callback] = [
        checkpoint_callback,
        early_stopping,
        ReporterProgressCallback(reporter, max_epochs=training_config.max_epochs),
    ]
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=training_config.max_epochs,
        callbacks=callbacks,
        deterministic=training_config.deterministic,
        gradient_clip_val=training_config.gradient_clip_norm,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=training_config.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(artifact_dir),
        precision=cast(Any, precision),
    )
    trainer.fit(module, datamodule=data_module)

    if checkpoint_callback.best_model_path:
        state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
    trained_model = _unwrap_compiled_model(module.model)
    if trained_model is not model:
        model.load_state_dict(trained_model.state_dict())
    best_epoch = _best_epoch(module.validation_history)
    return TrainingResult(
        best_epoch=best_epoch,
        train_history=module.train_history,
        validation_history=module.validation_history,
        best_checkpoint_path=(
            Path(checkpoint_callback.best_model_path)
            if checkpoint_callback.best_model_path
            else None
        ),
        resolved_precision=precision,
        compiled=compile_enabled,
        resolved_device=device.type,
    )


def evaluate_model(
    model: torch.nn.Module,
    *,
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    lookback_steps: int,
    training_config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
    reporter: Reporter | None = None,
) -> EpochMetrics:
    reporter = reporter or NullReporter()
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    device = resolve_device(training_config.device)
    model.to(device)
    model.eval()
    if class_weights is None:
        class_weights = build_class_weights(store.class_labels, sample_indices, store.action_count)
    class_weights = class_weights.to(device)
    loader = build_sequence_loader(
        store,
        sample_indices,
        lookback_steps=lookback_steps,
        batch_size=training_config.batch_size,
    )
    task_id = reporter.start_task("evaluate model", total=len(loader), unit="batches")
    metrics = []
    with torch.no_grad():
        for batch in loader:
            device_batch = move_batch_to_device(batch, device)
            outputs = model(device_batch.inputs)
            _, batch_metrics = compute_temporal_batch_metrics(
                outputs,
                device_batch,
                class_weights=class_weights,
                training_config=training_config,
            )
            metrics.append(batch_metrics)
            reporter.update_task(task_id, advance=1)
    reporter.finish_task(task_id)
    return mean_metrics(metrics)

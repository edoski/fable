"""Training utilities backed by Lightning."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy.typing import NDArray

from ..core.config import TrainingConfig
from ..core.console import NullReporter, Reporter
from ..data.datasets import TemporalDatasetStore
from ._runtime import (
    accumulation_steps as resolve_accumulation_steps,
)
from ._runtime import (
    build_sequence_loader,
    choose_microbatch_size,
    resolve_device,
    set_global_seed,
)
from .datamodule import TemporalDataModule
from .evaluation import EpochMetrics, compute_temporal_batch_metrics, mean_metrics
from .lightning_module import TemporalLightningModule
from .models import TemporalModel
from .torch_datasets import build_class_weights

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    train_history: list[EpochMetrics]
    validation_history: list[EpochMetrics]
    best_checkpoint_path: Path | None


@contextmanager
def _silence_lightning_logs() -> Iterator[None]:
    logger_names = (
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric",
        "lightning.fabric.utilities",
        "lightning.fabric.utilities.seed",
    )
    previous_levels: list[tuple[logging.Logger, int]] = []
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels.append((logger, logger.level))
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in previous_levels:
            logger.setLevel(level)


class _TrainingProgressCallback(L.Callback):
    def __init__(self, reporter: Reporter, task_id: int) -> None:
        self.reporter = reporter
        self.task_id = task_id

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        validation_loss = trainer.callback_metrics.get("validation_loss")
        validation_accuracy = trainer.callback_metrics.get("validation_accuracy")
        validation_profit = trainer.callback_metrics.get("validation/profit_over_baseline")
        parts: list[str] = []
        if validation_loss is not None:
            parts.append(f"loss={float(validation_loss):.4f}")
        if validation_accuracy is not None:
            parts.append(f"accuracy={float(validation_accuracy):.4f}")
        if validation_profit is not None:
            parts.append(f"profit={float(validation_profit):.4f}")
        self.reporter.update_task(
            self.task_id,
            completed=trainer.current_epoch + 1,
            message=" ".join(parts) if parts else None,
        )


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

    with _silence_lightning_logs():
        set_global_seed(training_config.seed)
        L.seed_everything(training_config.seed, workers=True)
    device = resolve_device(training_config.device)
    microbatch_size = choose_microbatch_size(training_config.batch_size, device)
    accumulation_steps = resolve_accumulation_steps(
        training_config.batch_size,
        microbatch_size,
    )

    data_module = TemporalDataModule(
        store=store,
        train_sample_indices=train_sample_indices,
        validation_sample_indices=validation_sample_indices,
        lookback_steps=lookback_steps,
        batch_size=training_config.batch_size,
        device=device,
    )

    module = TemporalLightningModule(
        model,
        class_weights=data_module.class_weights,
        action_count=store.action_count,
        training_config=training_config,
    )
    epoch_task_id = reporter.start_task(
        "train epochs",
        total=training_config.max_epochs,
        unit="epochs",
    )
    progress_callback = _TrainingProgressCallback(reporter, epoch_task_id)
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
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback, early_stopping, progress_callback],
        deterministic=training_config.deterministic,
        gradient_clip_val=training_config.gradient_clip_norm,
        accumulate_grad_batches=accumulation_steps,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=training_config.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(artifact_dir),
    )
    reporter.log(
        "training started "
        f"(accelerator={accelerator}, devices={devices}, microbatch={microbatch_size})"
    )
    with _silence_lightning_logs():
        trainer.fit(module, datamodule=data_module)

    if checkpoint_callback.best_model_path:
        state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
    reporter.finish_task(
        epoch_task_id,
        message=f"best_epoch={_best_epoch(module.validation_history)}",
    )
    reporter.log("training finished")
    return TrainingResult(
        best_epoch=_best_epoch(module.validation_history),
        train_history=module.train_history,
        validation_history=module.validation_history,
        best_checkpoint_path=(
            Path(checkpoint_callback.best_model_path)
            if checkpoint_callback.best_model_path
            else None
        ),
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
        device=device,
    )
    task_id = reporter.start_task("evaluate model", total=len(loader), unit="batches")
    metrics = []
    with torch.no_grad():
        for batch in loader:
            device_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(device_batch["inputs"])
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

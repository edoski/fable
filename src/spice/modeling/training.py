"""Training utilities backed by Lightning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy.typing import NDArray

from ..core.config import TrainingConfig
from ..core.console import ConsoleRuntime, NullReporter, Reporter
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
from .torch_datasets import build_class_weights, move_batch_to_device

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    train_history: list[EpochMetrics]
    validation_history: list[EpochMetrics]
    best_checkpoint_path: Path | None


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
    runtime: ConsoleRuntime | None = None,
) -> TrainingResult:
    reporter = reporter or NullReporter()
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

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
    progress_bar = None if runtime is None else runtime.lightning_progress_bar()
    callbacks: list[L.Callback] = [checkpoint_callback, early_stopping]
    if progress_bar is not None:
        callbacks.append(progress_bar)
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=training_config.max_epochs,
        callbacks=callbacks,
        deterministic=training_config.deterministic,
        gradient_clip_val=training_config.gradient_clip_norm,
        accumulate_grad_batches=accumulation_steps,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=progress_bar is not None,
        enable_model_summary=False,
        log_every_n_steps=training_config.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(artifact_dir),
    )
    reporter.log(
        "training started "
        f"(accelerator={accelerator}, devices={devices}, microbatch={microbatch_size})"
    )
    trainer.fit(module, datamodule=data_module)

    if checkpoint_callback.best_model_path:
        state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
    best_epoch = _best_epoch(module.validation_history)
    reporter.log(f"training finished: best_epoch={best_epoch}")
    return TrainingResult(
        best_epoch=best_epoch,
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

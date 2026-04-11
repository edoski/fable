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
from .evaluation import compute_batch_metrics
from .lightning_module import EpochMetrics, TemporalLightningModule, mean_metrics
from .models import TemporalModel
from .torch_datasets import build_class_weights

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
) -> TrainingResult:
    reporter = reporter or NullReporter()
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    set_global_seed(training_config.seed)
    L.seed_everything(training_config.seed, workers=True)
    device = resolve_device(training_config.device)
    microbatch_size = choose_microbatch_size(training_config.effective_batch_size, device)
    accumulation_steps = resolve_accumulation_steps(
        training_config.effective_batch_size,
        microbatch_size,
    )

    class_weights = build_class_weights(
        store.class_labels,
        train_sample_indices,
        store.action_count,
    )
    train_loader = build_sequence_loader(
        store,
        train_sample_indices,
        lookback_steps=lookback_steps,
        effective_batch_size=training_config.effective_batch_size,
        device=device,
    )
    validation_loader = build_sequence_loader(
        store,
        validation_sample_indices,
        lookback_steps=lookback_steps,
        effective_batch_size=training_config.effective_batch_size,
        device=device,
    )

    module = TemporalLightningModule(
        model,
        class_weights=class_weights,
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
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta,
    )
    accelerator, devices = _trainer_device_args(training_config.device)
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=training_config.deterministic,
        gradient_clip_val=training_config.gradient_clip_norm,
        accumulate_grad_batches=accumulation_steps,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=training_config.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(artifact_dir),
    )
    reporter.log(
        "training started "
        f"(accelerator={accelerator}, devices={devices}, microbatch={microbatch_size})"
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    if checkpoint_callback.best_model_path:
        state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
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
) -> EpochMetrics:
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")
    device = resolve_device(training_config.device)
    model.to(device)
    model.eval()
    if class_weights is None:
        class_weights = build_class_weights(store.class_labels, sample_indices, store.action_count)
    class_weights = class_weights.to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    smooth_l1 = torch.nn.SmoothL1Loss()
    loader = build_sequence_loader(
        store,
        sample_indices,
        lookback_steps=lookback_steps,
        effective_batch_size=training_config.effective_batch_size,
        device=device,
    )
    metrics = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            class_labels = batch["class_label"].to(device)
            target_log_fee = batch["target_log_fee"].to(device)
            outputs = model(inputs)
            block_loss = ce_loss(outputs.logits, class_labels)
            fee_loss = smooth_l1(outputs.fee_hat, target_log_fee)
            total_loss = training_config.alpha * block_loss + training_config.beta * fee_loss
            metrics.append(
                compute_batch_metrics(
                    logits=outputs.logits.detach(),
                    total_loss=total_loss.detach(),
                    class_labels=class_labels.detach(),
                    action_log_fees=batch["action_log_fees"].to(device).detach(),
                    next_block_log_fee=batch["next_block_log_fee"].to(device).detach(),
                    optimal_log_fee=batch["optimal_log_fee"].to(device).detach(),
                )
            )
    return mean_metrics(metrics)

"""Model training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from ..core.config import TrainingConfig
from ..core.console import NullReporter, Reporter
from ..data.datasets import TemporalDatasetStore
from ._runtime import (
    accumulation_steps as resolve_accumulation_steps,
)
from ._runtime import (
    build_sequence_loader,
    resolve_device,
    set_global_seed,
)
from .evaluation import BatchMetrics, compute_batch_metrics
from .models import ModelOutputs, TemporalModel
from .torch_datasets import SequenceBatch, build_class_weights

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class EpochMetrics:
    total_loss: float
    accuracy: float
    mean_cost_over_optimum: float
    mean_profit_over_baseline: float


@dataclass(slots=True)
class TrainingResult:
    best_epoch: int
    train_history: list[EpochMetrics]
    validation_history: list[EpochMetrics]


def _mean_metrics(metrics: list[BatchMetrics]) -> EpochMetrics:
    if not metrics:
        raise ValueError("Cannot summarize an empty metric list")
    denominator = sum(item.count for item in metrics)
    total_loss_sum = sum(item.total_loss_sum for item in metrics)
    correct_count = sum(item.correct_count for item in metrics)
    realized_fee_sum = sum(item.realized_fee_sum for item in metrics)
    baseline_fee_sum = sum(item.baseline_fee_sum for item in metrics)
    optimal_fee_sum = sum(item.optimal_fee_sum for item in metrics)
    return EpochMetrics(
        total_loss=total_loss_sum / denominator,
        accuracy=correct_count / denominator,
        mean_cost_over_optimum=(realized_fee_sum - optimal_fee_sum) / optimal_fee_sum,
        mean_profit_over_baseline=(baseline_fee_sum - realized_fee_sum) / baseline_fee_sum,
    )


def _run_epoch(
    model: TemporalModel,
    loader: DataLoader[SequenceBatch],
    *,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor,
    config: TrainingConfig,
    device: torch.device,
    accumulation_steps: int,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    smooth_l1 = nn.SmoothL1Loss()
    batch_metrics: list[BatchMetrics] = []

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        class_labels = batch["class_label"].to(device)
        target_log_fee = batch["target_log_fee"].to(device)
        action_log_fees = batch["action_log_fees"].to(device)
        next_block_log_fee = batch["next_block_log_fee"].to(device)
        optimal_log_fee = batch["optimal_log_fee"].to(device)

        with torch.set_grad_enabled(is_training):
            outputs: ModelOutputs = model(inputs)
            block_loss = ce_loss(outputs.logits, class_labels)
            fee_loss = smooth_l1(outputs.fee_hat, target_log_fee)
            total_loss = config.alpha * block_loss + config.beta * fee_loss

            if is_training:
                (total_loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_metrics.append(
            compute_batch_metrics(
                logits=outputs.logits.detach(),
                total_loss=total_loss.detach(),
                class_labels=class_labels.detach(),
                action_log_fees=action_log_fees.detach(),
                next_block_log_fee=next_block_log_fee.detach(),
                optimal_log_fee=optimal_log_fee.detach(),
            )
        )

    return _mean_metrics(batch_metrics)


def train_model(
    model: TemporalModel,
    *,
    store: TemporalDatasetStore,
    train_sample_indices: IntVector,
    validation_sample_indices: IntVector,
    lookback_steps: int,
    training_config: TrainingConfig,
    reporter: Reporter | None = None,
) -> TrainingResult:
    reporter = reporter or NullReporter()
    if train_sample_indices.size == 0 or validation_sample_indices.size == 0:
        raise ValueError("Train and validation sample selections must both be non-empty")

    set_global_seed(training_config.seed)
    device = resolve_device(training_config.device)
    model.to(device)
    class_weights = build_class_weights(
        store.class_labels,
        train_sample_indices,
        store.action_count,
    ).to(device)

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
    accumulation_steps = resolve_accumulation_steps(
        training_config.effective_batch_size,
        train_loader.batch_size or training_config.effective_batch_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_epoch = 1
    best_loss = float("inf")
    patience_left = training_config.early_stopping_patience
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    train_history: list[EpochMetrics] = []
    validation_history: list[EpochMetrics] = []
    reporter.start_training(total_epochs=training_config.max_epochs)

    for epoch in range(training_config.max_epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            config=training_config,
            device=device,
            accumulation_steps=accumulation_steps,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            class_weights=class_weights,
            config=training_config,
            device=device,
            accumulation_steps=1,
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

        improved = validation_metrics.total_loss < (
            best_loss - training_config.early_stopping_min_delta
        )
        if improved:
            best_loss = validation_metrics.total_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = training_config.early_stopping_patience
        else:
            patience_left -= 1

        reporter.training_epoch(
            epoch=epoch + 1,
            total_epochs=training_config.max_epochs,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_epoch=best_epoch,
            patience_left=patience_left,
        )
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    reporter.finish_training()
    return TrainingResult(
        best_epoch=best_epoch,
        train_history=train_history,
        validation_history=validation_history,
    )


def evaluate_model(
    model: TemporalModel,
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
    if class_weights is None:
        class_weights = build_class_weights(store.class_labels, sample_indices, store.action_count)
    class_weights = class_weights.to(device)
    loader = build_sequence_loader(
        store,
        sample_indices,
        lookback_steps=lookback_steps,
        effective_batch_size=training_config.effective_batch_size,
        device=device,
    )
    return _run_epoch(
        model,
        loader,
        optimizer=None,
        class_weights=class_weights,
        config=training_config,
        device=device,
        accumulation_steps=1,
    )

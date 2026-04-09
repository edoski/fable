"""Model training utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from spice_temporal.config import TrainingConfig
from spice_temporal.contracts import ModelOutputs, SequenceBatch
from spice_temporal.evaluation import BatchMetrics, compute_batch_metrics
from spice_temporal.records import SupervisedExample
from spice_temporal.torch_datasets import SequenceDataset, build_class_weights


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


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_microbatch_size(effective_batch_size: int, device: torch.device) -> int:
    candidates = [effective_batch_size, 32, 16, 8]
    if device.type == "cpu":
        return min(effective_batch_size, 16)
    for candidate in candidates:
        if candidate <= effective_batch_size:
            return candidate
    return 8


def _mean_metrics(metrics: list[BatchMetrics]) -> EpochMetrics:
    if not metrics:
        raise ValueError("Cannot summarize an empty metric list")
    denominator = sum(item.count for item in metrics)
    return EpochMetrics(
        total_loss=sum(item.total_loss * item.count for item in metrics) / denominator,
        accuracy=sum(item.accuracy * item.count for item in metrics) / denominator,
        mean_cost_over_optimum=(
            sum(item.mean_cost_over_optimum * item.count for item in metrics) / denominator
        ),
        mean_profit_over_baseline=(
            sum(item.mean_profit_over_baseline * item.count for item in metrics) / denominator
        ),
    )


def _run_epoch(
    model: nn.Module,
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
    optimizer_zeroed = False

    if is_training:
        optimizer.zero_grad(set_to_none=True)
        optimizer_zeroed = True

    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        class_labels = batch["class_label"].to(device)
        target_log_fee = batch["target_log_fee"].to(device)
        future_log_fees = batch["future_log_fees"].to(device)
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
                    optimizer_zeroed = True

        batch_metrics.append(
            compute_batch_metrics(
                logits=outputs.logits.detach(),
                total_loss=total_loss.detach(),
                class_labels=class_labels.detach(),
                future_log_fees=future_log_fees.detach(),
                next_block_log_fee=next_block_log_fee.detach(),
                optimal_log_fee=optimal_log_fee.detach(),
            )
        )

    if is_training and not optimizer_zeroed:
        optimizer.zero_grad(set_to_none=True)

    return _mean_metrics(batch_metrics)


def train_model(
    model: nn.Module,
    *,
    train_examples: Sequence[SupervisedExample],
    validation_examples: Sequence[SupervisedExample],
    config: TrainingConfig,
) -> TrainingResult:
    if not train_examples or not validation_examples:
        raise ValueError("Train and validation examples must both be non-empty")

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    n_classes = len(train_examples[0].future_log_fees)
    class_weights = build_class_weights(train_examples, n_classes).to(device)

    microbatch_size = choose_microbatch_size(config.effective_batch_size, device)
    accumulation_steps = max(1, math.ceil(config.effective_batch_size / microbatch_size))
    train_loader = DataLoader(
        SequenceDataset(train_examples),
        batch_size=microbatch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_examples),
        batch_size=microbatch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_epoch = 0
    best_loss = float("inf")
    patience_left = config.early_stopping_patience
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    train_history: list[EpochMetrics] = []
    validation_history: list[EpochMetrics] = []

    for epoch in range(config.max_epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            config=config,
            device=device,
            accumulation_steps=accumulation_steps,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            class_weights=class_weights,
            config=config,
            device=device,
            accumulation_steps=1,
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

        if validation_metrics.total_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_metrics.total_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        train_history=train_history,
        validation_history=validation_history,
    )


def evaluate_model(
    model: nn.Module,
    *,
    examples: Sequence[SupervisedExample],
    training_config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
) -> EpochMetrics:
    if not examples:
        raise ValueError("examples must be non-empty")
    device = resolve_device(training_config.device)
    model.to(device)
    n_classes = len(examples[0].future_log_fees)
    if class_weights is None:
        class_weights = build_class_weights(examples, n_classes)
    class_weights = class_weights.to(device)
    microbatch_size = choose_microbatch_size(training_config.effective_batch_size, device)
    loader = DataLoader(
        SequenceDataset(examples),
        batch_size=microbatch_size,
        shuffle=False,
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

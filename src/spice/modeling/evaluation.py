"""Evaluation helpers for training and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import TrainingConfig
from .models import ModelOutputs
from .torch_datasets import SequenceEventBatch


@dataclass(slots=True)
class BatchMetrics:
    count: int
    total_loss_sum: float
    correct_count: int
    realized_fee_sum: float
    baseline_fee_sum: float
    optimal_fee_sum: float


@dataclass(slots=True)
class EpochMetrics:
    total_loss: float
    accuracy: float
    mean_cost_over_optimum: float
    mean_profit_over_baseline: float


@dataclass(slots=True)
class TemporalLosses:
    action_loss: torch.Tensor
    fee_loss: torch.Tensor
    total_loss: torch.Tensor


def mean_metrics(metrics: list[BatchMetrics]) -> EpochMetrics:
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


def compute_temporal_losses(
    outputs: ModelOutputs,
    batch: SequenceEventBatch,
    *,
    class_weights: torch.Tensor,
    training_config: TrainingConfig,
) -> TemporalLosses:
    logits = outputs.logits.masked_fill(
        ~batch.candidate_mask,
        torch.finfo(outputs.logits.dtype).min,
    )
    fee_hat = outputs.fee_hat
    action_loss = F.cross_entropy(
        logits,
        batch.class_label,
        weight=class_weights.to(dtype=logits.dtype),
    )
    fee_loss = F.smooth_l1_loss(
        fee_hat,
        batch.target_log_fee.to(dtype=fee_hat.dtype),
    )
    total_loss = (
        training_config.action_loss_weight * action_loss
        + training_config.fee_loss_weight * fee_loss
    )
    return TemporalLosses(
        action_loss=action_loss,
        fee_loss=fee_loss,
        total_loss=total_loss,
    )


def compute_temporal_batch_metrics(
    outputs: ModelOutputs,
    batch: SequenceEventBatch,
    *,
    class_weights: torch.Tensor,
    training_config: TrainingConfig,
) -> tuple[torch.Tensor, BatchMetrics]:
    losses = compute_temporal_losses(
        outputs,
        batch,
        class_weights=class_weights,
        training_config=training_config,
    )
    return losses.total_loss, compute_batch_metrics(
        logits=outputs.logits.detach(),
        total_loss=losses.total_loss.detach(),
        class_labels=batch.class_label.detach(),
        candidate_log_fees=batch.candidate_log_fees.detach(),
        candidate_mask=batch.candidate_mask.detach(),
        next_block_log_fee=batch.next_block_log_fee.detach(),
        optimal_log_fee=batch.optimal_log_fee.detach(),
    )


def realized_log_fees_from_logits(
    logits: torch.Tensor,
    candidate_log_fees: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    predicted_offsets = masked_logits.argmax(dim=-1)
    return candidate_log_fees.gather(dim=1, index=predicted_offsets.unsqueeze(-1)).squeeze(-1)


def compute_batch_metrics(
    logits: torch.Tensor,
    total_loss: torch.Tensor,
    class_labels: torch.Tensor,
    candidate_log_fees: torch.Tensor,
    candidate_mask: torch.Tensor,
    next_block_log_fee: torch.Tensor,
    optimal_log_fee: torch.Tensor,
) -> BatchMetrics:
    count = class_labels.numel()
    masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    realized_log_fee = realized_log_fees_from_logits(
        masked_logits,
        candidate_log_fees,
        candidate_mask,
    )
    realized_fee = torch.exp(realized_log_fee)
    baseline_fee = torch.exp(next_block_log_fee)
    optimum_fee = torch.exp(optimal_log_fee)
    return BatchMetrics(
        count=count,
        total_loss_sum=total_loss.item() * count,
        correct_count=int((masked_logits.argmax(dim=-1) == class_labels).sum().item()),
        realized_fee_sum=float(realized_fee.sum().item()),
        baseline_fee_sum=float(baseline_fee.sum().item()),
        optimal_fee_sum=float(optimum_fee.sum().item()),
    )

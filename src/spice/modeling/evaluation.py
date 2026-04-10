"""Evaluation helpers for training and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BatchMetrics:
    count: int
    total_loss_sum: float
    correct_count: int
    realized_fee_sum: float
    baseline_fee_sum: float
    optimal_fee_sum: float


def realized_log_fees_from_logits(
    logits: torch.Tensor,
    action_log_fees: torch.Tensor,
) -> torch.Tensor:
    predicted_offsets = logits.argmax(dim=-1)
    return action_log_fees.gather(dim=1, index=predicted_offsets.unsqueeze(-1)).squeeze(-1)


def compute_batch_metrics(
    logits: torch.Tensor,
    total_loss: torch.Tensor,
    class_labels: torch.Tensor,
    action_log_fees: torch.Tensor,
    next_block_log_fee: torch.Tensor,
    optimal_log_fee: torch.Tensor,
) -> BatchMetrics:
    count = class_labels.numel()
    realized_log_fee = realized_log_fees_from_logits(logits, action_log_fees)
    realized_fee = torch.exp(realized_log_fee)
    baseline_fee = torch.exp(next_block_log_fee)
    optimum_fee = torch.exp(optimal_log_fee)
    return BatchMetrics(
        count=count,
        total_loss_sum=total_loss.item() * count,
        correct_count=int((logits.argmax(dim=-1) == class_labels).sum().item()),
        realized_fee_sum=float(realized_fee.sum().item()),
        baseline_fee_sum=float(baseline_fee.sum().item()),
        optimal_fee_sum=float(optimum_fee.sum().item()),
    )

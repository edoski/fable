"""Evaluation helpers for training and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BatchMetrics:
    total_loss: float
    accuracy: float
    mean_cost_over_optimum: float
    mean_profit_over_baseline: float


def realized_log_fees_from_logits(
    logits: torch.Tensor,
    future_log_fees: torch.Tensor,
) -> torch.Tensor:
    predicted_offsets = logits.argmax(dim=-1)
    gather_index = predicted_offsets.unsqueeze(-1)
    return future_log_fees.gather(dim=1, index=gather_index).squeeze(-1)


def compute_batch_metrics(
    logits: torch.Tensor,
    total_loss: torch.Tensor,
    class_labels: torch.Tensor,
    future_log_fees: torch.Tensor,
    next_block_log_fee: torch.Tensor,
    optimal_log_fee: torch.Tensor,
) -> BatchMetrics:
    realized_log_fee = realized_log_fees_from_logits(logits, future_log_fees)
    realized_fee = torch.exp(realized_log_fee)
    baseline_fee = torch.exp(next_block_log_fee)
    optimum_fee = torch.exp(optimal_log_fee)

    accuracy = (logits.argmax(dim=-1) == class_labels).float().mean().item()
    cost_over_optimum = ((realized_fee - optimum_fee) / optimum_fee.clamp_min(1e-8)).mean().item()
    profit_over_baseline = (
        (baseline_fee - realized_fee) / baseline_fee.clamp_min(1e-8)
    ).mean().item()
    return BatchMetrics(
        total_loss=total_loss.item(),
        accuracy=accuracy,
        mean_cost_over_optimum=cost_over_optimum,
        mean_profit_over_baseline=profit_over_baseline,
    )

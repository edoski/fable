"""Current-family metric derivation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...base import MetricDescriptor, MetricSet, WindowMetricSummary
from .batch import CandidateSlateTargetBatch
from .loss import compute_objective_loss
from .outputs import masked_candidate_logits


@dataclass(frozen=True, slots=True)
class CandidateSlateBatchState:
    count: int
    objective_loss_sum: float
    exact_hit_count: int
    realized_fee_sum: float
    baseline_fee_sum: float
    optimal_fee_sum: float


METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        id="profit_over_baseline",
        label="profit over baseline",
        role="primary",
    ),
    MetricDescriptor(
        id="cost_over_optimum",
        label="cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="objective_loss",
        label="objective loss",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="exact_optimum_hit_rate",
        label="exact optimum hit rate",
        role="diagnostic",
    ),
)


def _reference_tensors(
    logits: torch.Tensor,
    targets: CandidateSlateTargetBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = masked_candidate_logits(logits, targets.candidate_mask)
    masked_candidate_fees = targets.candidate_log_fees.masked_fill(
        ~targets.candidate_mask,
        torch.finfo(targets.candidate_log_fees.dtype).max,
    )
    predicted_candidate_index = masked_logits.argmax(dim=-1)
    optimal_candidate_index = masked_candidate_fees.argmin(dim=-1)
    realized_log_fee = targets.candidate_log_fees.gather(
        dim=1,
        index=predicted_candidate_index.unsqueeze(-1),
    ).squeeze(-1)
    optimal_log_fee = masked_candidate_fees.gather(
        dim=1,
        index=optimal_candidate_index.unsqueeze(-1),
    ).squeeze(-1)
    return predicted_candidate_index, optimal_candidate_index, realized_log_fee, optimal_log_fee


def compute_batch_loss_and_state(
    logits: torch.Tensor,
    targets: CandidateSlateTargetBatch,
) -> tuple[torch.Tensor, CandidateSlateBatchState]:
    objective_loss = compute_objective_loss(logits, targets)
    predicted_candidate_index, optimal_candidate_index, realized_log_fee, optimal_log_fee = (
        _reference_tensors(logits.detach(), targets)
    )
    realized_fee = torch.exp(realized_log_fee)
    baseline_fee = torch.exp(targets.candidate_log_fees[:, 0])
    optimum_fee = torch.exp(optimal_log_fee)
    count = int(targets.candidate_log_fees.shape[0])
    return objective_loss, CandidateSlateBatchState(
        count=count,
        objective_loss_sum=float(objective_loss.detach().item()) * count,
        exact_hit_count=int(
            (predicted_candidate_index == optimal_candidate_index).sum().item()
        ),
        realized_fee_sum=float(realized_fee.sum().item()),
        baseline_fee_sum=float(baseline_fee.sum().item()),
        optimal_fee_sum=float(optimum_fee.sum().item()),
    )


def summarize_epoch_metrics(batch_states: list[object]) -> MetricSet:
    states = [state for state in batch_states if isinstance(state, CandidateSlateBatchState)]
    if not states:
        raise ValueError("Cannot summarize an empty batch-state collection")
    count = sum(item.count for item in states)
    realized_fee_sum = sum(item.realized_fee_sum for item in states)
    baseline_fee_sum = sum(item.baseline_fee_sum for item in states)
    optimal_fee_sum = sum(item.optimal_fee_sum for item in states)
    return MetricSet(
        values={
            "objective_loss": sum(item.objective_loss_sum for item in states) / count,
            "exact_optimum_hit_rate": sum(item.exact_hit_count for item in states) / count,
            "cost_over_optimum": (realized_fee_sum - optimal_fee_sum) / optimal_fee_sum,
            "profit_over_baseline": (baseline_fee_sum - realized_fee_sum) / baseline_fee_sum,
        }
    )


def best_epoch(history: list[MetricSet]) -> int:
    if not history:
        return 1
    winner = max(
        range(len(history)),
        key=lambda index: (
            history[index].require("profit_over_baseline"),
            -history[index].require("cost_over_optimum"),
            -history[index].require("objective_loss"),
        ),
    )
    return winner + 1


def objective_value(metrics: MetricSet) -> float:
    return metrics.require("profit_over_baseline")


def summarize_window_metric(values: list[float]) -> WindowMetricSummary:
    return WindowMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )

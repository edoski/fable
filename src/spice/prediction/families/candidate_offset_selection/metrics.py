"""Candidate-offset selection metric derivation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...base import MetricDescriptor, MetricSet
from .batch import CandidateSlateTargetBatch
from .loss import compute_selection_loss
from .outputs import masked_candidate_logits


@dataclass(frozen=True, slots=True)
class CandidateSlateBatchState:
    count: int
    total_loss_sum: float
    exact_hit_count: int
    realized_fee_sum: float
    baseline_fee_sum: float
    optimal_fee_sum: float


@dataclass(slots=True)
class CandidateSlateEpochAccumulator:
    count: int = 0
    total_loss_sum: float = 0.0
    exact_hit_count: int = 0
    realized_fee_sum: float = 0.0
    baseline_fee_sum: float = 0.0
    optimal_fee_sum: float = 0.0

    def update(self, batch_state: object) -> None:
        if not isinstance(batch_state, CandidateSlateBatchState):
            raise TypeError("candidate_offset_selection expects CandidateSlateBatchState values")
        self.count += batch_state.count
        self.total_loss_sum += batch_state.total_loss_sum
        self.exact_hit_count += batch_state.exact_hit_count
        self.realized_fee_sum += batch_state.realized_fee_sum
        self.baseline_fee_sum += batch_state.baseline_fee_sum
        self.optimal_fee_sum += batch_state.optimal_fee_sum

    def snapshot(self) -> MetricSet:
        return _metric_set_from_totals(
            count=self.count,
            total_loss_sum=self.total_loss_sum,
            exact_hit_count=self.exact_hit_count,
            realized_fee_sum=self.realized_fee_sum,
            baseline_fee_sum=self.baseline_fee_sum,
            optimal_fee_sum=self.optimal_fee_sum,
        )

    def finalize(self) -> MetricSet:
        return self.snapshot()


TRAINING_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
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
        id="total_loss",
        label="total loss",
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
    total_loss = compute_selection_loss(logits, targets)
    predicted_candidate_index, optimal_candidate_index, realized_log_fee, optimal_log_fee = (
        _reference_tensors(logits.detach(), targets)
    )
    realized_fee = torch.exp(realized_log_fee)
    baseline_fee = torch.exp(targets.candidate_log_fees[:, 0])
    optimum_fee = torch.exp(optimal_log_fee)
    count = int(targets.candidate_log_fees.shape[0])
    return total_loss, CandidateSlateBatchState(
        count=count,
        total_loss_sum=float(total_loss.detach().item()) * count,
        exact_hit_count=int(
            (predicted_candidate_index == optimal_candidate_index).sum().item()
        ),
        realized_fee_sum=float(realized_fee.sum().item()),
        baseline_fee_sum=float(baseline_fee.sum().item()),
        optimal_fee_sum=float(optimum_fee.sum().item()),
    )


def create_epoch_accumulator() -> CandidateSlateEpochAccumulator:
    return CandidateSlateEpochAccumulator()


def best_epoch(history: list[MetricSet]) -> int:
    if not history:
        return 1
    winner = max(
        range(len(history)),
        key=lambda index: (
            history[index].require("profit_over_baseline"),
            -history[index].require("cost_over_optimum"),
            -history[index].require("total_loss"),
        ),
    )
    return winner + 1


def _metric_set_from_totals(
    *,
    count: int,
    total_loss_sum: float,
    exact_hit_count: int,
    realized_fee_sum: float,
    baseline_fee_sum: float,
    optimal_fee_sum: float,
) -> MetricSet:
    if count <= 0:
        raise ValueError("Cannot summarize an empty accumulator")
    return MetricSet(
        values={
            "total_loss": total_loss_sum / count,
            "exact_optimum_hit_rate": exact_hit_count / count,
            "cost_over_optimum": (realized_fee_sum - optimal_fee_sum) / optimal_fee_sum,
            "profit_over_baseline": (baseline_fee_sum - realized_fee_sum) / baseline_fee_sum,
        }
    )

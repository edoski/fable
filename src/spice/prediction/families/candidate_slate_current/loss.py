"""Current-family differentiable loss."""

from __future__ import annotations

import torch

from .batch import CandidateSlateTargetBatch
from .outputs import masked_candidate_logits


def compute_objective_loss(
    logits: torch.Tensor,
    targets: CandidateSlateTargetBatch,
) -> torch.Tensor:
    masked_logits = masked_candidate_logits(logits, targets.candidate_mask)
    policy = torch.softmax(masked_logits, dim=-1)
    baseline_log_fee = targets.candidate_log_fees[:, 0].unsqueeze(-1)
    relative_fee = torch.exp(targets.candidate_log_fees - baseline_log_fee)
    return (policy * relative_fee).sum(dim=-1).mean()

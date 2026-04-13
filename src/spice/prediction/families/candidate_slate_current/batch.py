"""Current-family target batch contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CandidateSlateTargetBatch:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor

    def to_device(self, device: torch.device) -> CandidateSlateTargetBatch:
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.to(device),
            candidate_mask=self.candidate_mask.to(device),
        )


@dataclass(frozen=True, slots=True)
class PreparedCandidateSlateTargets:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor

    def build_batch(self, sample_positions: torch.Tensor) -> CandidateSlateTargetBatch:
        positions = sample_positions.to(dtype=torch.int64)
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.index_select(0, positions),
            candidate_mask=self.candidate_mask.index_select(0, positions),
        )

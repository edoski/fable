"""Paper family target and training-state contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MinBlockFeeTargetBatch:
    candidate_mask: torch.Tensor
    min_block_offsets: torch.Tensor
    min_block_log_fees: torch.Tensor

    def to_device(self, device: torch.device) -> MinBlockFeeTargetBatch:
        return MinBlockFeeTargetBatch(
            candidate_mask=self.candidate_mask.to(device),
            min_block_offsets=self.min_block_offsets.to(device),
            min_block_log_fees=self.min_block_log_fees.to(device),
        )


@dataclass(frozen=True, slots=True)
class PreparedMinBlockFeeTargets:
    candidate_mask: torch.Tensor
    min_block_offsets: torch.Tensor
    min_block_log_fees: torch.Tensor

    def build_batch(self, sample_positions: torch.Tensor) -> MinBlockFeeTargetBatch:
        positions = sample_positions.to(dtype=torch.int64)
        return MinBlockFeeTargetBatch(
            candidate_mask=self.candidate_mask.index_select(0, positions),
            min_block_offsets=self.min_block_offsets.index_select(0, positions),
            min_block_log_fees=self.min_block_log_fees.index_select(0, positions),
        )


@dataclass(frozen=True, slots=True)
class MinBlockFeeTrainingState:
    class_weights: torch.Tensor
    fee_mean: float = 0.0
    fee_std: float = 1.0

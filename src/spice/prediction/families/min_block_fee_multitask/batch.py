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
        if (
            self.candidate_mask.device == device
            and self.min_block_offsets.device == device
            and self.min_block_log_fees.device == device
        ):
            return self
        non_blocking = device.type == "cuda"
        return MinBlockFeeTargetBatch(
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
            min_block_offsets=self.min_block_offsets.to(device, non_blocking=non_blocking),
            min_block_log_fees=self.min_block_log_fees.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> MinBlockFeeTargetBatch:
        if self.candidate_mask.device.type != "cpu":
            return self
        return MinBlockFeeTargetBatch(
            candidate_mask=self.candidate_mask.pin_memory(),
            min_block_offsets=self.min_block_offsets.pin_memory(),
            min_block_log_fees=self.min_block_log_fees.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class PreparedMinBlockFeeTargets:
    candidate_mask: torch.Tensor
    min_block_offsets: torch.Tensor
    min_block_log_fees: torch.Tensor
    storage_mode_id: str = "materialized_host"

    @property
    def estimated_storage_bytes(self) -> int:
        return (
            self.candidate_mask.element_size() * self.candidate_mask.numel()
            + self.min_block_offsets.element_size() * self.min_block_offsets.numel()
            + self.min_block_log_fees.element_size() * self.min_block_log_fees.numel()
        )

    def build_batch(self, sample_positions: torch.Tensor) -> MinBlockFeeTargetBatch:
        positions = sample_positions.detach().cpu().to(dtype=torch.int64, copy=False)
        index = positions.to(device=self.candidate_mask.device)
        return MinBlockFeeTargetBatch(
            candidate_mask=self.candidate_mask.index_select(0, index),
            min_block_offsets=self.min_block_offsets.index_select(0, index),
            min_block_log_fees=self.min_block_log_fees.index_select(0, index),
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedMinBlockFeeTargets | None:
        if self.candidate_mask.device == device:
            return self
        non_blocking = device.type == "cuda"
        return PreparedMinBlockFeeTargets(
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
            min_block_offsets=self.min_block_offsets.to(device, non_blocking=non_blocking),
            min_block_log_fees=self.min_block_log_fees.to(
                device,
                non_blocking=non_blocking,
            ),
            storage_mode_id="materialized_device",
        )


@dataclass(frozen=True, slots=True)
class MinBlockFeeTrainingState:
    class_weights: torch.Tensor
    fee_mean: float = 0.0
    fee_std: float = 1.0

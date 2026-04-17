"""Current-family target batch contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CandidateSlateTargetBatch:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor

    def to_device(self, device: torch.device) -> CandidateSlateTargetBatch:
        if self.candidate_log_fees.device == device and self.candidate_mask.device == device:
            return self
        non_blocking = device.type == "cuda"
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.to(device, non_blocking=non_blocking),
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> CandidateSlateTargetBatch:
        if self.candidate_log_fees.device.type != "cpu":
            return self
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.pin_memory(),
            candidate_mask=self.candidate_mask.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class PreparedCandidateSlateTargets:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor
    storage_mode_id: str = "materialized_host"

    @property
    def estimated_storage_bytes(self) -> int:
        return (
            self.candidate_log_fees.element_size() * self.candidate_log_fees.numel()
            + self.candidate_mask.element_size() * self.candidate_mask.numel()
        )

    def build_batch(self, sample_positions: torch.Tensor) -> CandidateSlateTargetBatch:
        positions = sample_positions.detach().cpu().to(dtype=torch.int64, copy=False)
        index = positions.to(device=self.candidate_log_fees.device)
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.index_select(0, index),
            candidate_mask=self.candidate_mask.index_select(0, index),
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedCandidateSlateTargets | None:
        if self.candidate_log_fees.device == device and self.candidate_mask.device == device:
            return self
        non_blocking = device.type == "cuda"
        return PreparedCandidateSlateTargets(
            candidate_log_fees=self.candidate_log_fees.to(device, non_blocking=non_blocking),
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
            storage_mode_id="materialized_device",
        )

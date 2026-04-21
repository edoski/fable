"""Current-family target batch contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ....temporal.problem_store import CompiledProblemStore
from ....temporal.realization import CompiledRealizationPolicyContract


def materialize_candidate_slate_targets(
    store: CompiledProblemStore,
    sample_indices: np.ndarray,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedCandidateSlateTargets:
    supervised = realization_policy.prepare_supervised_targets(
        store,
        sample_indices.astype(np.int64, copy=False),
    )
    return PreparedCandidateSlateTargets(
        candidate_log_fees=torch.from_numpy(supervised.candidate_log_fees),
        candidate_mask=torch.from_numpy(supervised.candidate_mask),
        optimum_offsets=torch.from_numpy(supervised.optimum_offsets),
        optimum_log_fees=torch.from_numpy(supervised.optimum_log_fees),
        baseline_candidate_indices=torch.from_numpy(supervised.baseline_candidate_indices),
    )


@dataclass(slots=True)
class CandidateSlateTargetBatch:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor
    optimum_offsets: torch.Tensor
    optimum_log_fees: torch.Tensor
    baseline_candidate_indices: torch.Tensor

    def to_device(self, device: torch.device) -> CandidateSlateTargetBatch:
        if (
            self.candidate_log_fees.device == device
            and self.candidate_mask.device == device
            and self.optimum_offsets.device == device
            and self.optimum_log_fees.device == device
            and self.baseline_candidate_indices.device == device
        ):
            return self
        non_blocking = device.type == "cuda"
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.to(device, non_blocking=non_blocking),
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
            optimum_offsets=self.optimum_offsets.to(device, non_blocking=non_blocking),
            optimum_log_fees=self.optimum_log_fees.to(device, non_blocking=non_blocking),
            baseline_candidate_indices=self.baseline_candidate_indices.to(
                device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> CandidateSlateTargetBatch:
        if self.candidate_log_fees.device.type != "cpu":
            return self
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.pin_memory(),
            candidate_mask=self.candidate_mask.pin_memory(),
            optimum_offsets=self.optimum_offsets.pin_memory(),
            optimum_log_fees=self.optimum_log_fees.pin_memory(),
            baseline_candidate_indices=self.baseline_candidate_indices.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class PreparedCandidateSlateTargets:
    candidate_log_fees: torch.Tensor
    candidate_mask: torch.Tensor
    optimum_offsets: torch.Tensor
    optimum_log_fees: torch.Tensor
    baseline_candidate_indices: torch.Tensor

    @property
    def estimated_storage_bytes(self) -> int:
        return (
            self.candidate_log_fees.element_size() * self.candidate_log_fees.numel()
            + self.candidate_mask.element_size() * self.candidate_mask.numel()
            + self.optimum_offsets.element_size() * self.optimum_offsets.numel()
            + self.optimum_log_fees.element_size() * self.optimum_log_fees.numel()
            + self.baseline_candidate_indices.element_size()
            * self.baseline_candidate_indices.numel()
        )

    def build_batch(self, sample_positions: torch.Tensor) -> CandidateSlateTargetBatch:
        positions = sample_positions.detach().cpu().to(dtype=torch.int64, copy=False)
        index = positions.to(device=self.candidate_log_fees.device)
        return CandidateSlateTargetBatch(
            candidate_log_fees=self.candidate_log_fees.index_select(0, index),
            candidate_mask=self.candidate_mask.index_select(0, index),
            optimum_offsets=self.optimum_offsets.index_select(0, index),
            optimum_log_fees=self.optimum_log_fees.index_select(0, index),
            baseline_candidate_indices=self.baseline_candidate_indices.index_select(0, index),
        )

    def to_device_storage(
        self,
        device: torch.device,
    ) -> PreparedCandidateSlateTargets:
        if (
            self.candidate_log_fees.device == device
            and self.candidate_mask.device == device
            and self.optimum_offsets.device == device
            and self.optimum_log_fees.device == device
            and self.baseline_candidate_indices.device == device
        ):
            return self
        non_blocking = device.type == "cuda"
        return PreparedCandidateSlateTargets(
            candidate_log_fees=self.candidate_log_fees.to(device, non_blocking=non_blocking),
            candidate_mask=self.candidate_mask.to(device, non_blocking=non_blocking),
            optimum_offsets=self.optimum_offsets.to(device, non_blocking=non_blocking),
            optimum_log_fees=self.optimum_log_fees.to(device, non_blocking=non_blocking),
            baseline_candidate_indices=self.baseline_candidate_indices.to(
                device, non_blocking=non_blocking
            ),
        )

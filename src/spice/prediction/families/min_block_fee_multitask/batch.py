"""Paper family target and training-state contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ....temporal.problem_store import CompiledProblemStore
from ....temporal.realization import CompiledRealizationPolicyContract


def materialize_min_block_fee_targets(
    store: CompiledProblemStore,
    sample_indices: np.ndarray,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedMinBlockFeeTargets:
    supervised = realization_policy.prepare_supervised_targets(
        store,
        sample_indices.astype(np.int64, copy=False),
    )
    return PreparedMinBlockFeeTargets(
        candidate_mask=torch.from_numpy(supervised.candidate_mask),
        min_block_offsets=torch.from_numpy(supervised.optimum_offsets),
        min_block_log_fees=torch.from_numpy(supervised.optimum_log_fees),
    )


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
    ) -> PreparedMinBlockFeeTargets:
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
        )


@dataclass(frozen=True, slots=True)
class ResolvedMinBlockFeeTrainingState:
    class_weights: torch.Tensor
    fee_mean: torch.Tensor
    fee_std: torch.Tensor


@dataclass(slots=True)
class MinBlockFeeTrainingState:
    class_weights: torch.Tensor
    fee_mean: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0.0, dtype=torch.float32)
    )
    fee_std: torch.Tensor = field(
        default_factory=lambda: torch.tensor(1.0, dtype=torch.float32)
    )
    _resolved: dict[tuple[str, int | None, torch.dtype], ResolvedMinBlockFeeTrainingState] = (
        field(default_factory=dict, init=False, repr=False)
    )

    def __post_init__(self) -> None:
        class_weights = self.class_weights.detach().to(device="cpu", dtype=torch.float32).clone()
        fee_mean = torch.as_tensor(self.fee_mean, dtype=torch.float32).detach().to(device="cpu")
        fee_std = torch.as_tensor(self.fee_std, dtype=torch.float32).detach().to(device="cpu")
        if fee_mean.ndim != 0 or fee_std.ndim != 0:
            raise ValueError("fee_mean and fee_std must be scalar tensors")
        if float(fee_std.item()) <= 0.0:
            raise ValueError("fee_std must be positive")
        self.class_weights = class_weights
        self.fee_mean = fee_mean.reshape(()).clone()
        self.fee_std = fee_std.reshape(()).clone()

    def resolve(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ResolvedMinBlockFeeTrainingState:
        key = (device.type, device.index, dtype)
        resolved = self._resolved.get(key)
        if resolved is not None:
            return resolved
        non_blocking = device.type == "cuda"
        resolved = ResolvedMinBlockFeeTrainingState(
            class_weights=self.class_weights.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            fee_mean=self.fee_mean.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            fee_std=self.fee_std.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
        )
        self._resolved[key] = resolved
        return resolved

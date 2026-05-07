# pyright: strict

"""Generic decoded-result ABI and decode context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

IntVector = NDArray[np.int64]
BoolMatrix = NDArray[np.bool_]


def coerce_cpu_int64_vector(
    values: torch.Tensor | IntVector,
    *,
    label: str,
) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return torch.as_tensor(values.astype(np.int64, copy=False), dtype=torch.int64)
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    return values.detach().to(device="cpu", dtype=torch.int64)


def _coerce_bool_matrix(
    values: torch.Tensor | BoolMatrix,
    *,
    label: str,
) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        if values.ndim != 2:
            raise ValueError(f"{label} must be two-dimensional")
        return torch.as_tensor(values.astype(np.bool_, copy=False), dtype=torch.bool)
    if values.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    return values.detach().to(dtype=torch.bool)


@runtime_checkable
class DecodedPredictionResult(Protocol):
    @property
    def decoded_result_id(self) -> str: ...

    def __len__(self) -> int: ...


class DecodeInputBatch(Protocol):
    @property
    def sample_positions(self) -> torch.Tensor: ...

    @property
    def action_mask(self) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ActionSpaceDecodeContext:
    sample_positions: torch.Tensor
    action_mask: torch.Tensor

    def __post_init__(self) -> None:
        sample_positions = coerce_cpu_int64_vector(
            self.sample_positions,
            label="sample_positions",
        )
        action_mask = _coerce_bool_matrix(
            self.action_mask,
            label="action_mask",
        )
        if sample_positions.shape[0] != action_mask.shape[0]:
            raise ValueError("sample_positions and action_mask must have matching rows")
        if action_mask.numel() > 0 and bool(torch.any(~action_mask.any(dim=1))):
            raise ValueError("action_mask must allow at least one action per sample")
        object.__setattr__(self, "sample_positions", sample_positions)
        object.__setattr__(self, "action_mask", action_mask)


def decode_context_from_batch(
    batch: DecodeInputBatch,
) -> ActionSpaceDecodeContext:
    return ActionSpaceDecodeContext(
        sample_positions=batch.sample_positions,
        action_mask=batch.action_mask,
    )

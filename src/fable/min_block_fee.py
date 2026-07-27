"""Architecture-neutral minimum-block-fee target, loss, and decode contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from pydantic import Field

from .records import StrictFrozenRecord

_FiniteFloat: TypeAlias = Annotated[
    float,
    Field(allow_inf_nan=False),
]
_PositiveFloat: TypeAlias = Annotated[
    float,
    Field(gt=0.0, allow_inf_nan=False),
]


class TargetState(StrictFrozenRecord):
    mean: _FiniteFloat
    standard_deviation: _PositiveFloat


class MinBlockFeeOutput(NamedTuple):
    action_logits: torch.Tensor
    minimum_fee_z: torch.Tensor


@dataclass(frozen=True, slots=True)
class MinBlockFeeLoss:
    mean_total: torch.Tensor
    total_by_origin: torch.Tensor


def _natural_log(values: NDArray[np.int64]) -> NDArray[np.float64]:
    return np.log(values.astype(np.float64, copy=False))


def fit_target_state(raw_minima: NDArray[np.int64]) -> TargetState:
    natural_log = _natural_log(raw_minima)
    mean = float(natural_log.mean(dtype=np.float64))
    standard_deviation = float(natural_log.std(dtype=np.float64, ddof=0))
    return TargetState(mean=mean, standard_deviation=standard_deviation)


def standardize_target(
    raw_minima: NDArray[np.int64],
    state: TargetState,
) -> NDArray[np.float32]:
    standardized = (_natural_log(raw_minima) - state.mean) / state.standard_deviation
    result = np.ascontiguousarray(standardized, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("standardized targets must be finite")
    return result


def min_block_fee_loss(
    output: MinBlockFeeOutput,
    *,
    label: torch.Tensor,
    target: torch.Tensor,
) -> MinBlockFeeLoss:
    classification = F.cross_entropy(
        output.action_logits,
        label,
        reduction="none",
    )
    regression = F.smooth_l1_loss(
        output.minimum_fee_z,
        target,
        reduction="none",
    )
    total = classification + regression
    return MinBlockFeeLoss(
        mean_total=total.sum() / output.action_logits.shape[0],
        total_by_origin=total.detach(),
    )


def decode_action(output: MinBlockFeeOutput) -> torch.Tensor:
    action_logits = output.action_logits
    if not torch.isfinite(action_logits).all():
        raise ValueError("action_logits must be finite")
    return action_logits.argmax(dim=-1)

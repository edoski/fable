"""Paper family output contract."""

from __future__ import annotations

import torch

from ...base import PredictionHeadSpec, PredictionOutputSpec

OFFSET_LOGITS_HEAD_ID = "min_block_offset_logits"
MIN_LOG_FEE_HEAD_ID = "min_block_log_fee"


def build_output_spec(max_candidate_slots: int) -> PredictionOutputSpec:
    if max_candidate_slots <= 0:
        raise ValueError("max_candidate_slots must be positive")
    return PredictionOutputSpec(
        heads=(
            PredictionHeadSpec(id=OFFSET_LOGITS_HEAD_ID, size=max_candidate_slots),
            PredictionHeadSpec(id=MIN_LOG_FEE_HEAD_ID, size=1),
        )
    )


def masked_offset_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if logits.shape != action_mask.shape:
        raise ValueError(
            f"logits and action_mask shapes must match: {logits.shape} != {action_mask.shape}"
        )
    if action_mask.ndim != 2:
        raise ValueError("action_mask must be two-dimensional")
    if not torch.all(action_mask.any(dim=-1)):
        raise ValueError("action_mask must allow at least one candidate per row")
    return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def masked_offset_argmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return masked_offset_logits(logits, action_mask).argmax(dim=-1)

"""Current-family output contract."""

from __future__ import annotations

import torch

from ...base import PredictionHeadSpec, PredictionOutputSpec

CANDIDATE_LOGITS_HEAD_ID = "candidate_logits"


def build_output_spec(max_candidate_slots: int) -> PredictionOutputSpec:
    if max_candidate_slots <= 0:
        raise ValueError("max_candidate_slots must be positive")
    return PredictionOutputSpec(
        heads=(PredictionHeadSpec(id=CANDIDATE_LOGITS_HEAD_ID, size=max_candidate_slots),)
    )


def candidate_logits(outputs) -> torch.Tensor:
    return outputs.head(CANDIDATE_LOGITS_HEAD_ID)


def masked_candidate_logits(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)

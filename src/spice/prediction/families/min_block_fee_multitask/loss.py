"""Paper-family multitask loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batch import MinBlockFeeTargetBatch, MinBlockFeeTrainingState
from .outputs import masked_offset_logits


def compute_multitask_loss(
    offset_logits: torch.Tensor,
    fee_predictions: torch.Tensor,
    targets: MinBlockFeeTargetBatch,
    *,
    training_state: MinBlockFeeTrainingState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = masked_offset_logits(offset_logits, targets.action_mask)
    resolved_state = training_state.resolve(
        device=masked_logits.device,
        dtype=masked_logits.dtype,
    )
    classification_loss = F.cross_entropy(
        masked_logits,
        targets.min_block_offsets,
        weight=resolved_state.class_weights,
    )
    normalized_state = training_state.resolve(
        device=fee_predictions.device,
        dtype=fee_predictions.dtype,
    )
    regression_targets = (
        targets.min_block_log_fees - normalized_state.fee_mean
    ) / normalized_state.fee_std
    regression_loss = F.smooth_l1_loss(
        fee_predictions,
        regression_targets,
    )
    total_loss = classification_loss + 0.5 * regression_loss
    return total_loss, classification_loss, regression_loss

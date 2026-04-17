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
    classification_loss_weight: float,
    regression_loss_weight: float,
    fee_target_normalization: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = masked_offset_logits(offset_logits, targets.candidate_mask)
    class_weights = training_state.class_weights.to(
        device=masked_logits.device,
        dtype=masked_logits.dtype,
    )
    classification_loss = F.cross_entropy(
        masked_logits,
        targets.min_block_offsets,
        weight=class_weights,
    )
    regression_targets = targets.min_block_log_fees
    if fee_target_normalization == "zscore_train_split":
        fee_mean = torch.tensor(
            training_state.fee_mean,
            device=fee_predictions.device,
            dtype=fee_predictions.dtype,
        )
        fee_std = torch.tensor(
            training_state.fee_std,
            device=fee_predictions.device,
            dtype=fee_predictions.dtype,
        )
        regression_targets = (regression_targets - fee_mean) / fee_std
    elif fee_target_normalization != "none":
        raise ValueError(f"Unsupported fee_target_normalization: {fee_target_normalization}")
    regression_loss = F.smooth_l1_loss(
        fee_predictions,
        regression_targets,
    )
    total_loss = (
        classification_loss_weight * classification_loss
        + regression_loss_weight * regression_loss
    )
    return total_loss, classification_loss, regression_loss

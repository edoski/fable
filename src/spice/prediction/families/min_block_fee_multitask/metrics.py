"""Paper-family training metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...base import MetricDescriptor, MetricSet
from .batch import MinBlockFeeTargetBatch, MinBlockFeeTrainingState
from .loss import compute_multitask_loss
from .outputs import masked_offset_logits


@dataclass(frozen=True, slots=True)
class MinBlockFeeBatchState:
    count: int
    total_loss_sum: float
    classification_loss_sum: float
    regression_loss_sum: float
    correct_offset_count: int


TRAINING_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(id="total_loss", label="total loss", role="primary"),
    MetricDescriptor(
        id="offset_accuracy",
        label="offset accuracy",
        role="secondary",
    ),
    MetricDescriptor(
        id="classification_loss",
        label="classification loss",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="regression_loss",
        label="regression loss",
        role="diagnostic",
    ),
)


def compute_batch_loss_and_state(
    offset_logits: torch.Tensor,
    fee_predictions: torch.Tensor,
    targets: MinBlockFeeTargetBatch,
    *,
    training_state: MinBlockFeeTrainingState,
    classification_loss_weight: float,
    regression_loss_weight: float,
) -> tuple[torch.Tensor, MinBlockFeeBatchState]:
    total_loss, classification_loss, regression_loss = compute_multitask_loss(
        offset_logits,
        fee_predictions,
        targets,
        training_state=training_state,
        classification_loss_weight=classification_loss_weight,
        regression_loss_weight=regression_loss_weight,
    )
    decoded_offsets = masked_offset_logits(
        offset_logits.detach(),
        targets.candidate_mask,
    ).argmax(dim=-1)
    count = int(targets.min_block_offsets.shape[0])
    return total_loss, MinBlockFeeBatchState(
        count=count,
        total_loss_sum=float(total_loss.detach().item()) * count,
        classification_loss_sum=float(classification_loss.detach().item()) * count,
        regression_loss_sum=float(regression_loss.detach().item()) * count,
        correct_offset_count=int((decoded_offsets == targets.min_block_offsets).sum().item()),
    )


def summarize_epoch_metrics(batch_states: list[object]) -> MetricSet:
    states = [state for state in batch_states if isinstance(state, MinBlockFeeBatchState)]
    if not states:
        raise ValueError("Cannot summarize an empty batch-state collection")
    count = sum(item.count for item in states)
    return MetricSet(
        values={
            "total_loss": sum(item.total_loss_sum for item in states) / count,
            "classification_loss": sum(item.classification_loss_sum for item in states) / count,
            "regression_loss": sum(item.regression_loss_sum for item in states) / count,
            "offset_accuracy": sum(item.correct_offset_count for item in states) / count,
        }
    )


def best_epoch(history: list[MetricSet]) -> int:
    if not history:
        return 1
    winner = min(
        range(len(history)),
        key=lambda index: (
            history[index].require("total_loss"),
            -history[index].require("offset_accuracy"),
            history[index].require("classification_loss"),
            history[index].require("regression_loss"),
        ),
    )
    return winner + 1


def inverse_frequency_class_weights(
    offsets: np.ndarray,
    *,
    n_classes: int,
) -> MinBlockFeeTrainingState:
    if offsets.size == 0:
        raise ValueError("offsets must be non-empty")
    counts = np.bincount(offsets, minlength=n_classes).astype(np.float32, copy=False)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if present.any():
        weights[present] *= float(present.sum()) / float(weights[present].sum())
    return MinBlockFeeTrainingState(class_weights=torch.from_numpy(weights))

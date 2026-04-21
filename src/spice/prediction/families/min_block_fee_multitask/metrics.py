"""Paper-family training metrics."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(slots=True)
class MinBlockFeeEpochAccumulator:
    count: int = 0
    total_loss_sum: float = 0.0
    classification_loss_sum: float = 0.0
    regression_loss_sum: float = 0.0
    correct_offset_count: int = 0

    def update(self, batch_state: object) -> None:
        if not isinstance(batch_state, MinBlockFeeBatchState):
            raise TypeError("min_block_fee_multitask expects MinBlockFeeBatchState values")
        self.count += batch_state.count
        self.total_loss_sum += batch_state.total_loss_sum
        self.classification_loss_sum += batch_state.classification_loss_sum
        self.regression_loss_sum += batch_state.regression_loss_sum
        self.correct_offset_count += batch_state.correct_offset_count

    def snapshot(self) -> MetricSet:
        return _metric_set_from_totals(
            count=self.count,
            total_loss_sum=self.total_loss_sum,
            classification_loss_sum=self.classification_loss_sum,
            regression_loss_sum=self.regression_loss_sum,
            correct_offset_count=self.correct_offset_count,
        )

    def finalize(self) -> MetricSet:
        return self.snapshot()


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
) -> tuple[torch.Tensor, MinBlockFeeBatchState]:
    total_loss, classification_loss, regression_loss = compute_multitask_loss(
        offset_logits,
        fee_predictions,
        targets,
        training_state=training_state,
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


def create_epoch_accumulator() -> MinBlockFeeEpochAccumulator:
    return MinBlockFeeEpochAccumulator()


def inverse_frequency_class_weights(
    offsets: torch.Tensor,
    *,
    n_classes: int,
) -> torch.Tensor:
    if offsets.ndim != 1:
        raise ValueError("offsets must be one-dimensional")
    if offsets.numel() == 0:
        raise ValueError("offsets must be non-empty")
    counts = torch.bincount(
        offsets.detach().to(device="cpu", dtype=torch.int64),
        minlength=n_classes,
    ).to(dtype=torch.float32)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if bool(present.any()):
        weights[present] *= float(present.sum().item()) / float(weights[present].sum().item())
    return weights


def _metric_set_from_totals(
    *,
    count: int,
    total_loss_sum: float,
    classification_loss_sum: float,
    regression_loss_sum: float,
    correct_offset_count: int,
) -> MetricSet:
    if count <= 0:
        raise ValueError("Cannot summarize an empty accumulator")
    return MetricSet(
        values={
            "total_loss": total_loss_sum / count,
            "classification_loss": classification_loss_sum / count,
            "regression_loss": regression_loss_sum / count,
            "offset_accuracy": correct_offset_count / count,
        }
    )

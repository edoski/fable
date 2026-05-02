"""Paper-family training metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....metrics import MetricDescriptor, MetricSet
from .._offset_metrics import (
    OffsetClassificationCounts,
    add_offset_classification_counts,
    macro_f1_from_counts,
    offset_classification_counts,
)
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
    log_fee_absolute_error_sum: float
    log_fee_squared_error_sum: float
    offset_counts: OffsetClassificationCounts


@dataclass(slots=True)
class MinBlockFeeEpochAccumulator:
    count: int = 0
    total_loss_sum: float = 0.0
    classification_loss_sum: float = 0.0
    regression_loss_sum: float = 0.0
    correct_offset_count: int = 0
    log_fee_absolute_error_sum: float = 0.0
    log_fee_squared_error_sum: float = 0.0
    offset_counts: OffsetClassificationCounts | None = None

    def update(self, batch_state: object) -> None:
        if not isinstance(batch_state, MinBlockFeeBatchState):
            raise TypeError("min_block_fee_multitask expects MinBlockFeeBatchState values")
        self.count += batch_state.count
        self.total_loss_sum += batch_state.total_loss_sum
        self.classification_loss_sum += batch_state.classification_loss_sum
        self.regression_loss_sum += batch_state.regression_loss_sum
        self.correct_offset_count += batch_state.correct_offset_count
        self.log_fee_absolute_error_sum += batch_state.log_fee_absolute_error_sum
        self.log_fee_squared_error_sum += batch_state.log_fee_squared_error_sum
        self.offset_counts = add_offset_classification_counts(
            self.offset_counts,
            batch_state.offset_counts,
        )

    def snapshot(self) -> MetricSet:
        if self.offset_counts is None:
            raise ValueError("Cannot summarize an empty accumulator")
        return _metric_set_from_totals(
            count=self.count,
            total_loss_sum=self.total_loss_sum,
            classification_loss_sum=self.classification_loss_sum,
            regression_loss_sum=self.regression_loss_sum,
            correct_offset_count=self.correct_offset_count,
            log_fee_absolute_error_sum=self.log_fee_absolute_error_sum,
            log_fee_squared_error_sum=self.log_fee_squared_error_sum,
            offset_counts=self.offset_counts,
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
    MetricDescriptor(
        id="macro_f1",
        label="macro F1",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="log_fee_mae",
        label="log fee MAE",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="log_fee_mse",
        label="log fee MSE",
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
        targets.action_mask,
    ).argmax(dim=-1)
    resolved_state = training_state.resolve(
        device=fee_predictions.device,
        dtype=fee_predictions.dtype,
    )
    predicted_log_fees = fee_predictions.detach() * resolved_state.fee_std + resolved_state.fee_mean
    log_fee_errors = predicted_log_fees - targets.min_block_log_fees.to(
        device=predicted_log_fees.device,
        dtype=predicted_log_fees.dtype,
    )
    count = int(targets.min_block_offsets.shape[0])
    offset_counts = offset_classification_counts(
        decoded_offsets,
        targets.min_block_offsets,
        n_classes=int(targets.action_mask.shape[1]),
    )
    return total_loss, MinBlockFeeBatchState(
        count=count,
        total_loss_sum=float(total_loss.detach().item()) * count,
        classification_loss_sum=float(classification_loss.detach().item()) * count,
        regression_loss_sum=float(regression_loss.detach().item()) * count,
        correct_offset_count=offset_counts.correct_count,
        log_fee_absolute_error_sum=float(log_fee_errors.abs().sum().detach().item()),
        log_fee_squared_error_sum=float(log_fee_errors.square().sum().detach().item()),
        offset_counts=offset_counts,
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
    log_fee_absolute_error_sum: float,
    log_fee_squared_error_sum: float,
    offset_counts: OffsetClassificationCounts,
) -> MetricSet:
    if count <= 0:
        raise ValueError("Cannot summarize an empty accumulator")
    return MetricSet(
        values={
            "total_loss": total_loss_sum / count,
            "classification_loss": classification_loss_sum / count,
            "regression_loss": regression_loss_sum / count,
            "offset_accuracy": correct_offset_count / count,
            "macro_f1": macro_f1_from_counts(offset_counts),
            "log_fee_mae": log_fee_absolute_error_sum / count,
            "log_fee_mse": log_fee_squared_error_sum / count,
        }
    )

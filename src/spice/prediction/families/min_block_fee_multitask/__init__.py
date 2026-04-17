"""Paper-faithful min-block-fee multitask prediction family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ....core.reporting import StageMetricDescriptor
from ....modeling.models import ModelOutputs
from ....temporal.problem_store import CompiledProblemStore
from ...base import MetricSet, PredictionOutputSpec
from ...contracts import (
    CompiledPredictionContract,
    EpochMetricAccumulator,
    IntVector,
    PredictionTargetBatch,
    PreparedPredictionTargets,
)
from ...registry import PredictionFamilySpec, register_prediction_family_spec
from .batch import (
    MinBlockFeeTargetBatch,
    MinBlockFeeTrainingState,
)
from .config import MinBlockFeeMultitaskFamilyConfig
from .metrics import (
    TRAINING_METRIC_DESCRIPTORS,
    best_epoch,
    compute_batch_loss_and_state,
    create_epoch_accumulator,
    inverse_frequency_class_weights,
)
from .outputs import (
    MIN_LOG_FEE_HEAD_ID,
    OFFSET_LOGITS_HEAD_ID,
    build_output_spec,
    masked_offset_logits,
)
from .targets import prepare_min_block_fee_targets

PROGRESS_METRIC_DESCRIPTORS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="total_loss", label="loss", width=7),
    StageMetricDescriptor(id="offset_accuracy", label="hit", width=6),
    StageMetricDescriptor(id="classification_loss", label="cls", width=7),
    StageMetricDescriptor(id="regression_loss", label="reg", width=7),
)


@dataclass(frozen=True, slots=True)
class MinBlockFeeMultitaskPredictionContract(CompiledPredictionContract):
    classification_loss_weight: float
    regression_loss_weight: float
    class_weighting: str
    fee_target_normalization: str

    def build_output_spec(self, max_candidate_slots: int) -> PredictionOutputSpec:
        return build_output_spec(max_candidate_slots)

    def fit_training_state(
        self,
        store: CompiledProblemStore,
        train_sample_indices: IntVector,
    ) -> object | None:
        if self.class_weighting != "inverse_frequency":
            raise ValueError(f"Unsupported class_weighting: {self.class_weighting}")
        targets = prepare_min_block_fee_targets(store, train_sample_indices)
        offsets = targets.min_block_offsets.detach().cpu().numpy().astype(np.int64, copy=False)
        state = inverse_frequency_class_weights(offsets, n_classes=store.max_candidate_slots)
        fee_mean = 0.0
        fee_std = 1.0
        if self.fee_target_normalization == "zscore_train_split":
            fees = targets.min_block_log_fees.detach().cpu().numpy().astype(np.float64, copy=False)
            fee_mean = float(fees.mean())
            fee_std = float(fees.std() + 1e-8)
        elif self.fee_target_normalization != "none":
            raise ValueError(
                f"Unsupported fee_target_normalization: {self.fee_target_normalization}"
            )
        return MinBlockFeeTrainingState(
            class_weights=state.class_weights,
            fee_mean=fee_mean,
            fee_std=fee_std,
        )

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedPredictionTargets:
        return prepare_min_block_fee_targets(store, sample_indices)

    def compute_batch_loss_and_state(
        self,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
        *,
        training_state: object | None,
    ) -> tuple[torch.Tensor, object]:
        if not isinstance(targets, MinBlockFeeTargetBatch):
            raise TypeError("min_block_fee_multitask expects MinBlockFeeTargetBatch targets")
        if not isinstance(training_state, MinBlockFeeTrainingState):
            raise TypeError("min_block_fee_multitask requires fitted MinBlockFeeTrainingState")
        return compute_batch_loss_and_state(
            outputs.head(OFFSET_LOGITS_HEAD_ID),
            outputs.head(MIN_LOG_FEE_HEAD_ID).squeeze(-1),
            targets,
            training_state=training_state,
            classification_loss_weight=self.classification_loss_weight,
            regression_loss_weight=self.regression_loss_weight,
            fee_target_normalization=self.fee_target_normalization,
        )

    def create_epoch_accumulator(self, stage: str) -> EpochMetricAccumulator:
        del stage
        return create_epoch_accumulator()

    def best_epoch(self, history: list[MetricSet]) -> int:
        return best_epoch(history)

    def allocate_decoded_offsets(self, sample_count: int) -> object:
        return [0] * sample_count

    def decode_selected_offsets_into(
        self,
        predictions: object,
        sample_positions: torch.Tensor,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        if not isinstance(targets, MinBlockFeeTargetBatch):
            raise TypeError("min_block_fee_multitask expects MinBlockFeeTargetBatch targets")
        if not isinstance(predictions, list):
            raise TypeError("min_block_fee_multitask decoded_offsets buffer must be a list")
        logits = masked_offset_logits(outputs.head(OFFSET_LOGITS_HEAD_ID), targets.candidate_mask)
        decoded = logits.argmax(dim=-1).cpu().tolist()
        positions = sample_positions.tolist()
        for sample_position, prediction in zip(positions, decoded, strict=True):
            predictions[int(sample_position)] = int(prediction)


def _compile(
    prediction_id: str,
    family: MinBlockFeeMultitaskFamilyConfig,
) -> CompiledPredictionContract:
    return MinBlockFeeMultitaskPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="min_block_fee_multitask",
        training_metric_descriptors=TRAINING_METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="total_loss",
        direction="minimize",
        supported_workflows=frozenset({"train", "tune", "evaluate"}),
        classification_loss_weight=family.classification_loss_weight,
        regression_loss_weight=family.regression_loss_weight,
        class_weighting=family.class_weighting,
        fee_target_normalization=family.fee_target_normalization,
    )


register_prediction_family_spec(
    PredictionFamilySpec(
        id="min_block_fee_multitask",
        config_type=MinBlockFeeMultitaskFamilyConfig,
        compile=_compile,
    )
)

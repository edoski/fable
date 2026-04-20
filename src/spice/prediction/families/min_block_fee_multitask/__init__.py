"""Min-block-fee multitask prediction family."""

from __future__ import annotations

import numpy as np
import torch

from ....core.reporting import StageMetricDescriptor
from ....modeling.models import ModelOutputs
from ....temporal.problem_store import CompiledProblemStore
from ....temporal.realization import CompiledRealizationPolicyContract
from ...contracts import (
    CompiledPredictionContract,
    DecodedOffsets,
    IntVector,
    PredictionTargetBatch,
    PreparedPredictionTargets,
)
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
)
from .targets import prepare_min_block_fee_targets

PROGRESS_METRIC_DESCRIPTORS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="total_loss", label="loss"),
    StageMetricDescriptor(id="offset_accuracy", label="hit"),
    StageMetricDescriptor(id="classification_loss", label="cls"),
    StageMetricDescriptor(id="regression_loss", label="reg"),
)


def _fit_training_state(
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    realization_policy: CompiledRealizationPolicyContract,
    *,
    class_weighting: str,
    fee_target_normalization: str,
) -> MinBlockFeeTrainingState:
    if class_weighting != "inverse_frequency":
        raise ValueError(f"Unsupported class_weighting: {class_weighting}")
    targets = prepare_min_block_fee_targets(
        store,
        train_sample_indices,
        realization_policy=realization_policy,
    )
    offsets = targets.min_block_offsets.detach().cpu().numpy().astype(np.int64, copy=False)
    state = inverse_frequency_class_weights(offsets, n_classes=store.max_candidate_slots)
    fee_mean = 0.0
    fee_std = 1.0
    if fee_target_normalization == "zscore_train_split":
        fees = targets.min_block_log_fees.detach().cpu().numpy().astype(np.float64, copy=False)
        fee_mean = float(fees.mean())
        fee_std = float(fees.std() + 1e-8)
    elif fee_target_normalization != "none":
        raise ValueError(f"Unsupported fee_target_normalization: {fee_target_normalization}")
    return MinBlockFeeTrainingState(
        class_weights=state.class_weights,
        fee_mean=fee_mean,
        fee_std=fee_std,
    )


def _prepare_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedPredictionTargets:
    return prepare_min_block_fee_targets(
        store,
        sample_indices,
        realization_policy=realization_policy,
    )


def _compute_batch_loss_and_state(
    outputs: ModelOutputs,
    targets: PredictionTargetBatch,
    training_state: object | None,
    *,
    classification_loss_weight: float,
    regression_loss_weight: float,
    fee_target_normalization: str,
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
        classification_loss_weight=classification_loss_weight,
        regression_loss_weight=regression_loss_weight,
        fee_target_normalization=fee_target_normalization,
    )


def _create_epoch_accumulator(stage: str):
    del stage
    return create_epoch_accumulator()


def _allocate_decoded_offsets(sample_count: int) -> DecodedOffsets:
    return [0] * sample_count


def _decode_selected_offsets_into(
    predictions: DecodedOffsets,
    sample_positions: torch.Tensor,
    outputs: ModelOutputs,
) -> None:
    decoded = outputs.head(OFFSET_LOGITS_HEAD_ID).argmax(dim=-1).cpu().tolist()
    positions = sample_positions.tolist()
    for sample_position, prediction in zip(positions, decoded, strict=True):
        predictions[int(sample_position)] = int(prediction)


def compile_prediction_family(
    prediction_id: str,
    family: MinBlockFeeMultitaskFamilyConfig,
) -> CompiledPredictionContract:
    return CompiledPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="min_block_fee_multitask",
        training_metric_descriptors=TRAINING_METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="total_loss",
        direction="minimize",
        supported_workflows=frozenset({"train", "tune", "evaluate"}),
        build_output_spec_fn=build_output_spec,
        fit_training_state_fn=lambda store, indices, realization_policy: _fit_training_state(
            store,
            indices,
            realization_policy,
            class_weighting=family.class_weighting,
            fee_target_normalization=family.fee_target_normalization,
        ),
        prepare_targets_fn=_prepare_targets,
        compute_batch_loss_and_state_fn=lambda outputs, targets, training_state: (
            _compute_batch_loss_and_state(
                outputs,
                targets,
                training_state,
                classification_loss_weight=family.classification_loss_weight,
                regression_loss_weight=family.regression_loss_weight,
                fee_target_normalization=family.fee_target_normalization,
            )
        ),
        create_epoch_accumulator_fn=_create_epoch_accumulator,
        select_best_epoch_fn=best_epoch,
        allocate_decoded_offsets_fn=_allocate_decoded_offsets,
        decode_selected_offsets_into_fn=_decode_selected_offsets_into,
    )

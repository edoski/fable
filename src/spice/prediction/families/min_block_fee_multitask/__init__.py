"""Min-block-fee multitask prediction family."""

from __future__ import annotations

import torch

from ....modeling.models import ModelOutputs
from ....temporal.execution_policy import CompiledExecutionPolicyContract, PreparedActionSpace
from ....temporal.problem_store import CompiledProblemStore
from ...contracts import (
    CompiledPredictionContract,
    IntVector,
    PredictionTargetBatch,
    PreparedPredictionTargets,
)
from ...decoded_offsets import DecodedOffsets, masked_offset_argmax
from ...decoding import ActionSpaceDecodeContext, DecodedPredictionResult
from .batch import (
    MinBlockFeeTargetBatch,
    MinBlockFeeTrainingState,
    materialize_min_block_fee_targets,
)
from .metrics import (
    TRAINING_METRIC_DESCRIPTORS,
    compute_batch_loss_and_state,
    create_epoch_accumulator,
    inverse_frequency_class_weights,
)
from .outputs import (
    MIN_LOG_FEE_HEAD_ID,
    OFFSET_LOGITS_HEAD_ID,
    build_output_spec,
)


def _fit_training_state(
    store: CompiledProblemStore,
    train_sample_indices: IntVector,
    execution_policy: CompiledExecutionPolicyContract,
) -> MinBlockFeeTrainingState:
    targets = materialize_min_block_fee_targets(
        store,
        train_sample_indices,
        execution_policy=execution_policy,
        action_space=execution_policy.prepare_action_space(store, train_sample_indices),
    )
    class_weights = inverse_frequency_class_weights(
        targets.min_block_offsets,
        n_classes=store.max_candidate_slots,
    )
    fees = targets.min_block_log_fees.detach().to(device="cpu", dtype=torch.float32)
    fee_mean = fees.mean()
    fee_std = fees.std(correction=0) + 1e-8
    return MinBlockFeeTrainingState(
        class_weights=class_weights,
        fee_mean=fee_mean,
        fee_std=fee_std,
    )


def _prepare_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    execution_policy: CompiledExecutionPolicyContract,
    action_space: PreparedActionSpace,
) -> PreparedPredictionTargets:
    return materialize_min_block_fee_targets(
        store,
        sample_indices,
        execution_policy=execution_policy,
        action_space=action_space,
    )


def _compute_batch_loss_and_state(
    outputs: ModelOutputs,
    targets: PredictionTargetBatch,
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
    )


def _create_epoch_accumulator():
    return create_epoch_accumulator()


def _decode_candidate_offsets_into(
    predictions: DecodedPredictionResult,
    outputs: ModelOutputs,
    decode_context: ActionSpaceDecodeContext,
) -> None:
    if not isinstance(predictions, DecodedOffsets):
        raise TypeError("min_block_fee_multitask decodes candidate offsets")
    predictions.write(
        decode_context.sample_positions,
        masked_offset_argmax(
            outputs.head(OFFSET_LOGITS_HEAD_ID),
            decode_context.action_mask,
        ),
    )


def compile_prediction_family(
    prediction_id: str,
) -> CompiledPredictionContract:
    return CompiledPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="min_block_fee_multitask",
        training_metric_descriptors=TRAINING_METRIC_DESCRIPTORS,
        primary_metric_id="total_loss",
        direction="minimize",
        build_output_spec_fn=build_output_spec,
        fit_training_state_fn=_fit_training_state,
        prepare_targets_fn=_prepare_targets,
        compute_batch_loss_and_state_fn=_compute_batch_loss_and_state,
        create_epoch_accumulator_fn=_create_epoch_accumulator,
        decoded_result_id=DecodedOffsets.decoded_result_id,
        allocate_decoded_result_fn=DecodedOffsets.allocate,
        decode_batch_result_into_fn=_decode_candidate_offsets_into,
    )

"""Candidate-offset selection prediction family."""

from __future__ import annotations

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
from .batch import CandidateSlateTargetBatch
from .config import CandidateOffsetSelectionFamilyConfig
from .metrics import (
    TRAINING_METRIC_DESCRIPTORS,
    best_epoch,
    compute_batch_loss_and_state,
    create_epoch_accumulator,
)
from .outputs import (
    CANDIDATE_LOGITS_HEAD_ID,
    build_output_spec,
    candidate_logits,
)
from .targets import prepare_candidate_slate_targets

PROGRESS_METRIC_DESCRIPTORS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="profit_over_baseline", label="profit"),
    StageMetricDescriptor(id="cost_over_optimum", label="cost"),
    StageMetricDescriptor(id="total_loss", label="loss"),
    StageMetricDescriptor(id="exact_optimum_hit_rate", label="hit"),
)


def _prepare_targets(
    store: CompiledProblemStore,
    sample_indices: IntVector,
    realization_policy: CompiledRealizationPolicyContract,
) -> PreparedPredictionTargets:
    return prepare_candidate_slate_targets(
        store,
        sample_indices,
        realization_policy=realization_policy,
    )


def _compute_batch_loss_and_state(
    outputs: ModelOutputs,
    targets: PredictionTargetBatch,
    training_state: object | None,
) -> tuple[torch.Tensor, object]:
    del training_state
    if not isinstance(targets, CandidateSlateTargetBatch):
        raise TypeError("candidate_offset_selection expects CandidateSlateTargetBatch targets")
    return compute_batch_loss_and_state(outputs.head(CANDIDATE_LOGITS_HEAD_ID), targets)


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
    decoded = candidate_logits(outputs).argmax(dim=-1).cpu().tolist()
    positions = sample_positions.tolist()
    for sample_position, prediction in zip(positions, decoded, strict=True):
        predictions[int(sample_position)] = int(prediction)


def compile_prediction_family(
    prediction_id: str,
    family: CandidateOffsetSelectionFamilyConfig,
) -> CompiledPredictionContract:
    del family
    return CompiledPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="candidate_offset_selection",
        training_metric_descriptors=TRAINING_METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        supported_workflows=frozenset({"train", "tune", "evaluate"}),
        build_output_spec_fn=build_output_spec,
        prepare_targets_fn=_prepare_targets,
        compute_batch_loss_and_state_fn=_compute_batch_loss_and_state,
        create_epoch_accumulator_fn=_create_epoch_accumulator,
        select_best_epoch_fn=best_epoch,
        allocate_decoded_offsets_fn=_allocate_decoded_offsets,
        decode_selected_offsets_into_fn=_decode_selected_offsets_into,
    )

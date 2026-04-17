"""Candidate-offset selection prediction family."""

from __future__ import annotations

from dataclasses import dataclass

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
    masked_candidate_logits,
)
from .targets import prepare_candidate_slate_targets

PROGRESS_METRIC_DESCRIPTORS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="profit_over_baseline", label="profit", width=8),
    StageMetricDescriptor(id="cost_over_optimum", label="cost", width=8),
    StageMetricDescriptor(id="total_loss", label="loss", width=7),
    StageMetricDescriptor(id="exact_optimum_hit_rate", label="hit", width=6),
)


@dataclass(frozen=True, slots=True)
class CandidateOffsetSelectionPredictionContract(CompiledPredictionContract):
    def build_output_spec(self, max_candidate_slots: int) -> PredictionOutputSpec:
        return build_output_spec(max_candidate_slots)

    def prepare_targets(
        self,
        store: CompiledProblemStore,
        sample_indices: IntVector,
    ) -> PreparedPredictionTargets:
        return prepare_candidate_slate_targets(store, sample_indices)

    def compute_batch_loss_and_state(
        self,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
        *,
        training_state: object | None,
    ) -> tuple[torch.Tensor, object]:
        del training_state
        if not isinstance(targets, CandidateSlateTargetBatch):
            raise TypeError("candidate_offset_selection expects CandidateSlateTargetBatch targets")
        return compute_batch_loss_and_state(outputs.head(CANDIDATE_LOGITS_HEAD_ID), targets)

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
        sample_positions,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        if not isinstance(targets, CandidateSlateTargetBatch):
            raise TypeError("candidate_offset_selection expects CandidateSlateTargetBatch targets")
        if not isinstance(predictions, list):
            raise TypeError("candidate_offset_selection decoded_offsets buffer must be a list")
        logits = masked_candidate_logits(candidate_logits(outputs), targets.candidate_mask)
        decoded = logits.argmax(dim=-1).cpu().tolist()
        positions = sample_positions.tolist()
        for sample_position, prediction in zip(positions, decoded, strict=True):
            predictions[int(sample_position)] = int(prediction)


def _compile(
    prediction_id: str,
    family: CandidateOffsetSelectionFamilyConfig,
) -> CompiledPredictionContract:
    del family
    return CandidateOffsetSelectionPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="candidate_offset_selection",
        training_metric_descriptors=TRAINING_METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        supported_workflows=frozenset({"train", "tune", "evaluate"}),
    )


register_prediction_family_spec(
    PredictionFamilySpec(
        id="candidate_offset_selection",
        config_type=CandidateOffsetSelectionFamilyConfig,
        compile=_compile,
    )
)

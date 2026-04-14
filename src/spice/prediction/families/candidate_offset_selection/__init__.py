"""Candidate-offset selection prediction family."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....core.reporting import Reporter, StageMetricDescriptor
from ....modeling.models import ModelOutputs
from ....temporal.problem_store import CompiledProblemStore
from ...base import MetricSet, PredictionOutputSpec, PredictionSimulationSummary
from ...contracts import (
    CompiledPredictionContract,
    IntVector,
    PredictionTargetBatch,
    PreparedPredictionTargets,
)
from ...registry import PredictionFamilySpec, register_prediction_family_spec
from .batch import CandidateSlateTargetBatch
from .config import CandidateOffsetSelectionFamilyConfig
from .metrics import (
    METRIC_DESCRIPTORS,
    best_epoch,
    compute_batch_loss_and_state,
    summarize_epoch_metrics,
)
from .outputs import CANDIDATE_LOGITS_HEAD_ID, build_output_spec
from .replay import allocate_prediction_buffer, decode_into, run_replay
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

    def summarize_epoch_metrics(self, batch_states: list[object]) -> MetricSet:
        return summarize_epoch_metrics(batch_states)

    def best_epoch(self, history: list[MetricSet]) -> int:
        return best_epoch(history)

    def allocate_prediction_buffer(self, sample_count: int) -> object:
        return allocate_prediction_buffer(sample_count)

    def decode_into(
        self,
        predictions: object,
        sample_positions,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        if not isinstance(targets, CandidateSlateTargetBatch):
            raise TypeError("candidate_offset_selection expects CandidateSlateTargetBatch targets")
        decode_into(predictions, sample_positions, outputs, targets)

    def replay(
        self,
        store: CompiledProblemStore,
        predictions: object,
        sample_indices: IntVector,
        window_seconds: int,
        arrival_rate_per_second: float,
        repetitions: int,
        seed: int,
        reporter: Reporter | None,
    ) -> PredictionSimulationSummary:
        return run_replay(
            store,
            predictions,
            sample_indices,
            window_seconds,
            arrival_rate_per_second,
            repetitions,
            seed,
            reporter,
        )


def _compile(
    prediction_id: str,
    family: CandidateOffsetSelectionFamilyConfig,
) -> CompiledPredictionContract:
    del family
    return CandidateOffsetSelectionPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="candidate_offset_selection",
        metric_descriptors=METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        supported_workflows=frozenset({"train", "tune", "simulate"}),
    )


register_prediction_family_spec(
    PredictionFamilySpec(
        id="candidate_offset_selection",
        config_type=CandidateOffsetSelectionFamilyConfig,
        compile=_compile,
    )
)

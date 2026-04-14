"""Paper-faithful min-block-fee multitask prediction family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
from .batch import (
    MinBlockFeeTargetBatch,
    MinBlockFeeTrainingState,
)
from .config import MinBlockFeeMultitaskFamilyConfig
from .metrics import (
    METRIC_DESCRIPTORS,
    best_epoch,
    compute_batch_loss_and_state,
    inverse_frequency_class_weights,
    summarize_epoch_metrics,
)
from .outputs import MIN_LOG_FEE_HEAD_ID, OFFSET_LOGITS_HEAD_ID, build_output_spec
from .replay import allocate_prediction_buffer, decode_into, run_replay
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
        return inverse_frequency_class_weights(offsets, n_classes=store.max_candidate_slots)

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
        )

    def summarize_epoch_metrics(self, batch_states: list[object]) -> MetricSet:
        return summarize_epoch_metrics(batch_states)

    def best_epoch(self, history: list[MetricSet]) -> int:
        return best_epoch(history)

    def allocate_prediction_buffer(self, sample_count: int) -> object:
        return allocate_prediction_buffer(sample_count)

    def decode_into(
        self,
        predictions: object,
        sample_positions: torch.Tensor,
        outputs: ModelOutputs,
        targets: PredictionTargetBatch,
    ) -> None:
        if not isinstance(targets, MinBlockFeeTargetBatch):
            raise TypeError("min_block_fee_multitask expects MinBlockFeeTargetBatch targets")
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
    family: MinBlockFeeMultitaskFamilyConfig,
) -> CompiledPredictionContract:
    return MinBlockFeeMultitaskPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="min_block_fee_multitask",
        metric_descriptors=METRIC_DESCRIPTORS,
        progress_metric_descriptors=PROGRESS_METRIC_DESCRIPTORS,
        primary_metric_id="total_loss",
        direction="minimize",
        supported_workflows=frozenset({"train", "tune", "simulate"}),
        classification_loss_weight=family.classification_loss_weight,
        regression_loss_weight=family.regression_loss_weight,
        class_weighting=family.class_weighting,
    )


register_prediction_family_spec(
    PredictionFamilySpec(
        id="min_block_fee_multitask",
        config_type=MinBlockFeeMultitaskFamilyConfig,
        compile=_compile,
    )
)

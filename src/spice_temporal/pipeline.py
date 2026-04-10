"""Training and inference dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spice_temporal.constants import EVALUATION_END_TS, EVALUATION_START_TS
from spice_temporal.datasets import (
    DatasetGeometry,
    DatasetSplitIndices,
    TemporalDatasetStore,
    build_temporal_store,
    chronological_split_indices,
    derive_dataset_geometry,
    filter_sample_indices_by_anchor_window,
    history_context_blocks,
    trim_history_blocks_for_target,
)
from spice_temporal.features import build_feature_table
from spice_temporal.io import load_block_records
from spice_temporal.models import TemporalModel, build_model
from spice_temporal.normalization import (
    StandardScaler,
    fit_standard_scaler,
    transform_feature_matrix,
)
from spice_temporal.records import BlockRecord
from spice_temporal.specs import TrainingSpec
from spice_temporal.torch_datasets import build_class_weights
from spice_temporal.training import EpochMetrics, TrainingResult, evaluate_model, train_model

IntVector = NDArray[np.int64]


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_blocks_available: int
    n_blocks_used: int
    n_examples_total: int
    store: TemporalDatasetStore
    split_indices: DatasetSplitIndices
    scaler: StandardScaler
    geometry: DatasetGeometry

    @property
    def n_features(self) -> int:
        return self.store.n_features

    @property
    def action_count(self) -> int:
        return self.store.action_count


@dataclass(slots=True)
class PreparedInferenceDataset:
    n_history_context_blocks: int
    n_evaluation_blocks: int
    n_examples_total: int
    store: TemporalDatasetStore
    sample_indices: IntVector
    geometry: DatasetGeometry

    @property
    def n_features(self) -> int:
        return self.store.n_features


@dataclass(slots=True)
class TrainingRunResult:
    model: TemporalModel
    prepared: PreparedTrainingDataset
    training_result: TrainingResult
    test_metrics: EpochMetrics


def prepare_training_dataset(
    blocks: list[BlockRecord],
    *,
    spec: TrainingSpec,
) -> PreparedTrainingDataset:
    geometry = derive_dataset_geometry(
        lookback_seconds=spec.lookback_seconds,
        max_delay_seconds=spec.max_delay_seconds,
        block_time_seconds=spec.chain.block_time_seconds,
    )
    trimmed_blocks = trim_history_blocks_for_target(
        blocks,
        target_anchor_count=spec.target_anchor_count,
        geometry=geometry,
    )
    feature_table = build_feature_table(trimmed_blocks)
    store = build_temporal_store(
        feature_table,
        lookback_steps=geometry.lookback_steps,
        action_count=geometry.action_count,
    )
    if store.n_samples != spec.target_anchor_count:
        raise RuntimeError(
            "Training dataset preparation produced an unexpected number of anchors; "
            f"expected {spec.target_anchor_count}, got {store.n_samples}"
        )

    split_indices = chronological_split_indices(store.n_samples, spec.split)
    scaler = fit_standard_scaler(
        store.feature_matrix,
        anchor_row_indices=store.anchor_row_indices,
        sample_indices=split_indices.train,
        lookback_steps=geometry.lookback_steps,
    )
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    return PreparedTrainingDataset(
        n_blocks_available=len(blocks),
        n_blocks_used=len(trimmed_blocks),
        n_examples_total=store.n_samples,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        geometry=geometry,
    )


def prepare_inference_dataset(
    history_blocks: list[BlockRecord],
    evaluation_blocks: list[BlockRecord],
    *,
    geometry: DatasetGeometry,
    scaler: StandardScaler,
) -> PreparedInferenceDataset:
    context_blocks = history_context_blocks(history_blocks, geometry=geometry)
    combined_blocks = [*context_blocks, *evaluation_blocks]
    feature_table = build_feature_table(combined_blocks)
    store = build_temporal_store(
        feature_table,
        lookback_steps=geometry.lookback_steps,
        action_count=geometry.action_count,
    )
    sample_indices = filter_sample_indices_by_anchor_window(
        store,
        start_timestamp=EVALUATION_START_TS,
        end_timestamp=EVALUATION_END_TS,
    )
    if sample_indices.size == 0:
        raise ValueError("Evaluation dataset produced no valid inference examples")

    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    return PreparedInferenceDataset(
        n_history_context_blocks=len(context_blocks),
        n_evaluation_blocks=len(evaluation_blocks),
        n_examples_total=int(sample_indices.shape[0]),
        store=scaled_store,
        sample_indices=sample_indices,
        geometry=geometry,
    )


def run_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
) -> TrainingRunResult:
    blocks = load_block_records(history_block_path)
    prepared = prepare_training_dataset(
        blocks,
        spec=spec,
    )
    model = build_model(prepared.n_features, prepared.action_count, spec.model)
    training_result = train_model(
        model,
        store=prepared.store,
        train_sample_indices=prepared.split_indices.train,
        validation_sample_indices=prepared.split_indices.validation,
        lookback_steps=prepared.geometry.lookback_steps,
        training_config=spec.training,
    )
    class_weights = build_class_weights(
        prepared.store.class_labels,
        prepared.split_indices.train,
        prepared.action_count,
    )
    test_metrics = evaluate_model(
        model,
        store=prepared.store,
        sample_indices=prepared.split_indices.test,
        lookback_steps=prepared.geometry.lookback_steps,
        training_config=spec.training,
        class_weights=class_weights,
    )
    return TrainingRunResult(
        model=model,
        prepared=prepared,
        training_result=training_result,
        test_metrics=test_metrics,
    )

"""Training and inference dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from ..core.config import ChainConfig, ModelConfig, SplitConfig, TrainingConfig
from ..core.console import Reporter
from ..data.datasets import (
    DatasetGeometry,
    DatasetSplitIndices,
    IntVector,
    TemporalDatasetStore,
    build_temporal_store,
    chronological_split_indices,
    derive_dataset_geometry,
    filter_sample_indices_by_anchor_window,
    history_context_slice,
    trim_history_for_anchor_count,
)
from ..data.features import build_feature_table
from ..data.io import load_enriched_block_frame
from ..data.normalization import ScalerStats, fit_standard_scaler, transform_feature_matrix
from .models import TemporalModel, build_model
from .torch_datasets import build_class_weights
from .training import EpochMetrics, TrainingResult, evaluate_model, train_model


@dataclass(slots=True)
class TrainingSpec:
    chain: ChainConfig
    dataset_id: str
    model: ModelConfig
    max_delay_seconds: int
    lookback_seconds: int
    target_anchor_count: int
    split: SplitConfig
    training: TrainingConfig


@dataclass(slots=True)
class PreparedTrainingDataset:
    n_blocks_available: int
    n_blocks_used: int
    n_examples_total: int
    store: TemporalDatasetStore
    split_indices: DatasetSplitIndices
    scaler: ScalerStats
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


def _slice_frame(frame: pl.DataFrame, selection: slice) -> pl.DataFrame:
    start = 0 if selection.start is None else selection.start
    stop = frame.height if selection.stop is None else selection.stop
    return frame.slice(start, stop - start)


def prepare_training_dataset(
    blocks: pl.DataFrame,
    *,
    spec: TrainingSpec,
) -> PreparedTrainingDataset:
    geometry = derive_dataset_geometry(
        lookback_seconds=spec.lookback_seconds,
        max_delay_seconds=spec.max_delay_seconds,
        block_time_seconds=spec.chain.block_time_seconds,
    )
    trimmed_blocks = _slice_frame(
        blocks.sort("block_number"),
        trim_history_for_anchor_count(
            blocks.height,
            anchor_count=spec.target_anchor_count,
            geometry=geometry,
        ),
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
        n_blocks_available=blocks.height,
        n_blocks_used=trimmed_blocks.height,
        n_examples_total=store.n_samples,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        geometry=geometry,
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    *,
    geometry: DatasetGeometry,
    scaler: ScalerStats,
    evaluation_start_timestamp: int,
    evaluation_end_timestamp: int,
) -> PreparedInferenceDataset:
    context_blocks = _slice_frame(
        history_blocks.sort("block_number"),
        history_context_slice(history_blocks.height, geometry=geometry),
    )
    combined_blocks = pl.concat([context_blocks, evaluation_blocks.sort("block_number")])
    feature_table = build_feature_table(combined_blocks)
    store = build_temporal_store(
        feature_table,
        lookback_steps=geometry.lookback_steps,
        action_count=geometry.action_count,
    )
    sample_indices = filter_sample_indices_by_anchor_window(
        store,
        start_timestamp=evaluation_start_timestamp,
        end_timestamp=evaluation_end_timestamp,
    )
    if sample_indices.size == 0:
        raise ValueError("Evaluation dataset produced no valid inference examples")

    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )
    return PreparedInferenceDataset(
        n_history_context_blocks=context_blocks.height,
        n_evaluation_blocks=evaluation_blocks.height,
        n_examples_total=int(sample_indices.shape[0]),
        store=scaled_store,
        sample_indices=sample_indices,
        geometry=geometry,
    )


def run_training(
    history_block_path: Path,
    *,
    spec: TrainingSpec,
    artifact_dir: Path,
    reporter: Reporter | None = None,
) -> TrainingRunResult:
    blocks = load_enriched_block_frame(history_block_path)
    prepared = prepare_training_dataset(blocks, spec=spec)
    model = build_model(prepared.n_features, prepared.action_count, spec.model)
    training_result = train_model(
        model,
        store=prepared.store,
        train_sample_indices=prepared.split_indices.train,
        validation_sample_indices=prepared.split_indices.validation,
        lookback_steps=prepared.geometry.lookback_steps,
        training_config=spec.training,
        artifact_dir=artifact_dir,
        reporter=reporter,
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

"""Fixed-context temporal dataset-preparation path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from ...core.errors import ConfigResolutionError
from ...temporal.problem_store import (
    CompiledProblemStore,
    DatasetSplitIndices,
    chronological_split_indices,
)
from ...temporal.scaling import transform_feature_matrix
from .base import (
    CompiledDatasetBuilderContract,
    FixedSequenceTemporalBuilderRuntimeMetadata,
    FixedSequenceTemporalDatasetBuilderConfig,
    fixed_sequence_temporal_runtime_metadata,
    validate_feature_prerequisites,
)
from .preparation import (
    CompiledInferenceDatasetPreparationSpec,
    PreparedInferenceDataset,
    PreparedTrainingDataset,
    TrainingDatasetPreparationSpec,
)


def _prepare_blocks(blocks: pl.DataFrame) -> pl.DataFrame:
    if blocks.height == 0:
        raise ValueError("dataset builder received an empty block frame")
    return (
        blocks.sort("timestamp")
        .unique(subset=["block_number"], keep="first", maintain_order=True)
        .sort("block_number")
    )


def _compute_seq_len(
    timestamps: np.ndarray,
    *,
    lookback_seconds: int,
    min_sequence_length: int,
    max_sequence_length: int,
) -> tuple[int, float]:
    if timestamps.size < 2:
        raise ValueError("fixed-sequence builder requires at least two timestamps")
    deltas = np.diff(timestamps.astype(np.float64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError("fixed-sequence builder requires positive timestamp deltas")
    median_dt = float(np.median(positive_deltas))
    seq_len = int(round(lookback_seconds / max(median_dt, 1e-6)))
    seq_len = int(np.clip(seq_len, min_sequence_length, max_sequence_length))
    return seq_len, median_dt


def _build_selected_store(
    blocks: pl.DataFrame,
    *,
    spec: TrainingDatasetPreparationSpec,
) -> tuple[CompiledProblemStore, object]:
    feature_table = spec.feature_contract.build_table(_prepare_blocks(blocks))
    validate_feature_prerequisites(
        feature_table.feature_prerequisites,
        spec.problem_contract.feature_prerequisites,
    )
    return spec.problem_contract.build_capability_store(feature_table)


def _training_calibration_timestamps(
    store: CompiledProblemStore,
    train_sample_indices: np.ndarray,
) -> np.ndarray:
    if train_sample_indices.size == 0:
        raise ValueError("fixed-sequence builder requires training samples")
    context = store.context_windows(train_sample_indices)
    start_row = int(context.context_start_rows.min())
    stop_row = int(context.anchor_rows.max()) + 1
    return store.timestamps[start_row:stop_row].astype(np.int64, copy=False)


def _train_scaler(
    store: CompiledProblemStore,
    *,
    spec: TrainingDatasetPreparationSpec,
    train_sample_indices: np.ndarray,
):
    return spec.input_normalization_contract.fit_scaler(
        store.feature_matrix,
        context_start_rows=store.context_start_rows,
        anchor_rows=store.anchor_rows,
        sample_indices=train_sample_indices.astype(np.int64, copy=False),
    )


def _split_store_indices(
    selected_sample_indices: np.ndarray,
    *,
    spec: TrainingDatasetPreparationSpec,
) -> DatasetSplitIndices:
    split_positions = chronological_split_indices(
        int(selected_sample_indices.shape[0]),
        spec.split,
    )
    return DatasetSplitIndices(
        train=selected_sample_indices[split_positions.train],
        validation=selected_sample_indices[split_positions.validation],
        test=selected_sample_indices[split_positions.test],
    )


def _scale_store(
    store: CompiledProblemStore,
    *,
    scaler,
) -> CompiledProblemStore:
    return replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )


def prepare_training_dataset(
    blocks: pl.DataFrame,
    spec: TrainingDatasetPreparationSpec,
) -> PreparedTrainingDataset:
    if not isinstance(spec.dataset_builder, FixedSequenceTemporalDatasetBuilderConfig):
        raise TypeError("fixed-sequence builder requires FixedSequenceTemporalDatasetBuilderConfig")
    sorted_blocks = _prepare_blocks(blocks)
    if sorted_blocks.height < spec.sample_count:
        raise ValueError(
            "History dataset is too short for the requested sample count; "
            f"need at least {spec.sample_count} rows, got {sorted_blocks.height}"
        )
    store_raw, compiler_runtime_metadata = _build_selected_store(sorted_blocks, spec=spec)
    raw_selected_sample_indices = store_raw.tail_sample_indices(
        sample_count=spec.sample_count,
    )
    raw_split_indices = _split_store_indices(
        raw_selected_sample_indices,
        spec=spec,
    )
    seq_len, median_dt_seconds = _compute_seq_len(
        _training_calibration_timestamps(store_raw, raw_split_indices.train),
        lookback_seconds=spec.lookback_seconds,
        min_sequence_length=spec.dataset_builder.min_sequence_length,
        max_sequence_length=spec.dataset_builder.max_sequence_length,
    )
    history_seconds = spec.problem_contract.feature_prerequisites.history_seconds
    warmup_rows = spec.problem_contract.feature_prerequisites.warmup_rows
    store = store_raw.with_fixed_context_length(
        context_length=seq_len,
        history_seconds=history_seconds,
        warmup_rows=warmup_rows,
    )
    selected_sample_indices = store.tail_sample_indices(
        sample_count=spec.sample_count,
    )
    split_indices = _split_store_indices(
        selected_sample_indices,
        spec=spec,
    )
    scaler = _train_scaler(
        store,
        spec=spec,
        train_sample_indices=split_indices.train,
    )
    scaled_store = _scale_store(store, scaler=scaler)
    used_start, used_end = store.selected_row_span(selected_sample_indices)
    return PreparedTrainingDataset(
        n_rows_available=sorted_blocks.height,
        n_rows_used=used_end - used_start,
        sample_count=int(selected_sample_indices.shape[0]),
        execution_policy=spec.problem_contract.execution_policy,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        builder_runtime_metadata=fixed_sequence_temporal_runtime_metadata(
            compiler_id=spec.problem_contract.compiler_id,
            compiler_runtime_metadata=compiler_runtime_metadata,
            sequence_length=int(seq_len),
            median_dt_seconds=float(median_dt_seconds),
            min_sequence_length=spec.dataset_builder.min_sequence_length,
            max_sequence_length=spec.dataset_builder.max_sequence_length,
        ),
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    spec: CompiledInferenceDatasetPreparationSpec,
) -> PreparedInferenceDataset:
    combined_blocks = _prepare_blocks(pl.concat([history_blocks, evaluation_blocks]))
    if not isinstance(spec.builder_runtime_metadata, FixedSequenceTemporalBuilderRuntimeMetadata):
        raise ConfigResolutionError(
            "fixed_sequence_temporal builder requires fixed_sequence_temporal runtime metadata"
        )
    runtime_metadata = spec.builder_runtime_metadata
    feature_table = spec.feature_contract.build_table(combined_blocks)
    store = spec.problem_contract.build_delay_store(
        feature_table,
        spec.delay_seconds,
        compiler_runtime_metadata=spec.compiler_runtime_metadata,
        max_candidate_slots=spec.max_candidate_slots,
    )
    store = store.with_fixed_context_length(
        context_length=runtime_metadata.sequence_length,
        history_seconds=spec.problem_contract.feature_prerequisites.history_seconds,
        warmup_rows=spec.problem_contract.feature_prerequisites.warmup_rows,
    )
    sample_indices = store.sample_indices_by_timestamp_window(
        start_timestamp=spec.window_start_timestamp,
        end_timestamp=spec.window_end_timestamp,
    )
    if sample_indices.size == 0:
        raise ValueError("Evaluation dataset produced no valid inference examples")
    scaled_store = replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, spec.scaler),
    )
    return PreparedInferenceDataset(
        n_history_rows=history_blocks.height,
        n_evaluation_rows=evaluation_blocks.height,
        sample_count=int(sample_indices.shape[0]),
        execution_policy=spec.problem_contract.execution_policy,
        store=scaled_store,
        sample_indices=sample_indices,
    )


def compile_dataset_builder(
    config: FixedSequenceTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id != "fixed_sequence_temporal":
        raise ConfigResolutionError("dataset_builder.id must be fixed_sequence_temporal")
    return CompiledDatasetBuilderContract(
        dataset_builder_id="fixed_sequence_temporal",
        prepare_training_fn=prepare_training_dataset,
        prepare_inference_fn=prepare_inference_dataset,
    )

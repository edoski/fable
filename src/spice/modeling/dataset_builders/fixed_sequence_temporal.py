"""Fixed-context temporal dataset-preparation path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from ...core.errors import ConfigResolutionError
from ...temporal.contracts import TemporalCapabilityStore
from ...temporal.problem_store import CompiledProblemStore
from ...temporal.scaling import transform_feature_matrix
from .base import (
    CompiledDatasetBuilderContract,
    FixedSequenceTemporalBuilderRuntimeMetadata,
    FixedSequenceTemporalDatasetBuilderConfig,
    fixed_sequence_temporal_runtime_metadata,
    validate_feature_prerequisites,
)
from .preparation import (
    CompiledInferenceDatasetPreparationRequest,
    DatasetSplitIndices,
    PreparedInferenceDataset,
    PreparedInferenceSampleSelection,
    PreparedTrainingDataset,
    PreparedTrainingSampleRoles,
    TrainingDatasetPreparationContext,
    TrainingDatasetPreparationFacts,
    training_sample_selection,
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
    context: TrainingDatasetPreparationContext,
) -> TemporalCapabilityStore:
    feature_table = context.feature_contract.build_table(_prepare_blocks(blocks))
    validate_feature_prerequisites(
        feature_table.feature_prerequisites,
        context.problem_contract.feature_prerequisites,
    )
    return context.problem_contract.build_capability_store(feature_table)


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
    context: TrainingDatasetPreparationContext,
    train_sample_indices: np.ndarray,
):
    return context.input_normalization_contract.fit_scaler(
        store.feature_matrix,
        context_start_rows=store.context_start_rows,
        anchor_rows=store.anchor_rows,
        sample_indices=train_sample_indices.astype(np.int64, copy=False),
    )


def _split_store_indices(
    selected_sample_indices: np.ndarray,
    *,
    facts: TrainingDatasetPreparationFacts,
) -> DatasetSplitIndices:
    split_positions = _chronological_split_indices(
        int(selected_sample_indices.shape[0]),
        facts.split,
    )
    return DatasetSplitIndices(
        train=selected_sample_indices[split_positions.train],
        validation=selected_sample_indices[split_positions.validation],
        test=selected_sample_indices[split_positions.test],
    )


def _chronological_split_indices(n_samples: int, split_config) -> DatasetSplitIndices:
    if n_samples < 3:
        raise ValueError("Need at least three examples to create train/validation/test splits")

    train_end = int(n_samples * split_config.train_fraction)
    validation_end = train_end + int(n_samples * split_config.validation_fraction)
    train_end = max(1, min(train_end, n_samples - 2))
    validation_end = max(train_end + 1, min(validation_end, n_samples - 1))
    all_indices = np.arange(n_samples, dtype=np.int64)
    return DatasetSplitIndices(
        train=all_indices[:train_end],
        validation=all_indices[train_end:validation_end],
        test=all_indices[validation_end:],
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
    facts: TrainingDatasetPreparationFacts,
    context: TrainingDatasetPreparationContext,
    config: FixedSequenceTemporalDatasetBuilderConfig,
) -> PreparedTrainingDataset:
    sorted_blocks = _prepare_blocks(blocks)
    sample_count = context.problem_contract.sample_count
    if sorted_blocks.height < sample_count:
        raise ValueError(
            "History dataset is too short for the requested sample count; "
            f"need at least {sample_count} rows, got {sorted_blocks.height}"
        )
    capability_store = _build_selected_store(sorted_blocks, context=context)
    store_raw = capability_store.store
    raw_selected_sample_indices = store_raw.tail_sample_indices(
        sample_count=sample_count,
    )
    raw_split_indices = _split_store_indices(
        raw_selected_sample_indices,
        facts=facts,
    )
    seq_len, median_dt_seconds = _compute_seq_len(
        _training_calibration_timestamps(store_raw, raw_split_indices.train),
        lookback_seconds=context.problem_contract.lookback_seconds,
        min_sequence_length=config.min_sequence_length,
        max_sequence_length=config.max_sequence_length,
    )
    history_seconds = context.problem_contract.feature_prerequisites.history_seconds
    warmup_rows = context.problem_contract.feature_prerequisites.warmup_rows
    store = store_raw.with_fixed_context_length(
        context_length=seq_len,
        history_seconds=history_seconds,
        warmup_rows=warmup_rows,
    )
    selected_sample_indices = store.tail_sample_indices(
        sample_count=sample_count,
    )
    split_indices = _split_store_indices(
        selected_sample_indices,
        facts=facts,
    )
    scaler = _train_scaler(
        store,
        context=context,
        train_sample_indices=split_indices.train,
    )
    scaled_store = _scale_store(store, scaler=scaler)
    execution_policy = context.problem_contract.execution_policy
    samples = PreparedTrainingSampleRoles(
        train=training_sample_selection(
            "train",
            execution_policy.prepare_temporal_facts(scaled_store, split_indices.train),
        ),
        validation=training_sample_selection(
            "validation",
            execution_policy.prepare_temporal_facts(
                scaled_store,
                split_indices.validation,
            ),
        ),
        test=training_sample_selection(
            "test",
            execution_policy.prepare_temporal_facts(scaled_store, split_indices.test),
        ),
    )
    used_start, used_end = store.selected_row_span(selected_sample_indices)
    return PreparedTrainingDataset(
        n_rows_available=sorted_blocks.height,
        n_rows_used=used_end - used_start,
        sample_count=int(selected_sample_indices.shape[0]),
        execution_policy=execution_policy,
        store=scaled_store,
        samples=samples,
        scaler=scaler,
        builder_runtime_metadata=fixed_sequence_temporal_runtime_metadata(
            sequence_length=int(seq_len),
            median_dt_seconds=float(median_dt_seconds),
            min_sequence_length=config.min_sequence_length,
            max_sequence_length=config.max_sequence_length,
        ),
        temporal_capability=capability_store.capability,
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    spec: CompiledInferenceDatasetPreparationRequest,
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
        capability=spec.temporal_capability,
    )
    store = store.with_fixed_context_length(
        context_length=runtime_metadata.sequence_length,
        history_seconds=spec.problem_contract.feature_prerequisites.history_seconds,
        warmup_rows=spec.problem_contract.feature_prerequisites.warmup_rows,
    )
    sample_indices = store.sample_indices_by_timestamp_window(
        start_timestamp_inclusive=(
            spec.sample_timestamp_window.start_timestamp_inclusive
        ),
        end_timestamp_exclusive=(
            spec.sample_timestamp_window.end_timestamp_exclusive
        ),
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
        execution_policy=spec.problem_contract.execution_policy,
        store=scaled_store,
        samples=PreparedInferenceSampleSelection(
            role="inference",
            action_space=spec.problem_contract.execution_policy.prepare_action_space(
                scaled_store,
                sample_indices,
            ),
        ),
    )


def compile_dataset_builder(
    config: FixedSequenceTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id != "fixed_sequence_temporal":
        raise ConfigResolutionError("dataset_builder.id must be fixed_sequence_temporal")
    return CompiledDatasetBuilderContract(
        dataset_builder_id="fixed_sequence_temporal",
        prepare_training_fn=lambda blocks, facts, context: prepare_training_dataset(
            blocks,
            facts,
            context,
            config,
        ),
        prepare_inference_fn=prepare_inference_dataset,
    )

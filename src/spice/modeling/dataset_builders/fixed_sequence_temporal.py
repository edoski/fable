"""Fixed-context temporal dataset-preparation path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from ...config.models import SequenceConfig
from ...temporal.contracts import TemporalCapabilityStore
from ...temporal.input_normalization import transform_problem_store_features
from ...temporal.problem_store import CompiledProblemStore
from .base import (
    sequence_runtime_metadata,
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
    train_sample_indices: np.ndarray,
):
    from ...temporal.input_normalization import fit_row_standard_scaler

    return fit_row_standard_scaler(
        store,
        sample_indices=train_sample_indices.astype(np.int64, copy=False),
    )


def _all_sample_indices(store: CompiledProblemStore) -> np.ndarray:
    return np.arange(store.n_samples, dtype=np.int64)


def _training_sample_indices(
    store: CompiledProblemStore,
    *,
    training_cutoff_timestamp: int | None,
) -> np.ndarray:
    sample_indices = _all_sample_indices(store)
    if training_cutoff_timestamp is None:
        return sample_indices
    outcome_end_rows = store.candidate_end_rows[sample_indices] - 1
    mask = store.timestamps[outcome_end_rows] < training_cutoff_timestamp
    selected = sample_indices[mask].astype(np.int64, copy=False)
    if selected.size == 0:
        raise ValueError(
            "training cutoff produced no valid supervised samples: "
            f"training_cutoff_timestamp={training_cutoff_timestamp}"
        )
    return selected


def _store_with_fixed_context(
    store: CompiledProblemStore,
    *,
    context_length: int,
    history_seconds: int,
    warmup_rows: int,
) -> CompiledProblemStore:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    context_start_rows = store.anchor_rows - context_length + 1
    valid_anchor_mask = context_start_rows >= 0
    valid_anchor_mask &= context_start_rows >= warmup_rows
    if history_seconds > 0:
        valid_anchor_mask &= (
            store.timestamps[np.maximum(context_start_rows, 0)] - store.timestamps[0]
        ) >= history_seconds
    anchor_rows = store.anchor_rows[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("fixed context length produced no supervised samples")
    return replace(
        store,
        anchor_rows=anchor_rows,
        context_start_rows=context_start_rows[valid_anchor_mask].astype(
            np.int64,
            copy=False,
        ),
        candidate_start_rows=store.candidate_start_rows[valid_anchor_mask].astype(
            np.int64,
            copy=False,
        ),
        candidate_end_rows=store.candidate_end_rows[valid_anchor_mask].astype(
            np.int64,
            copy=False,
        ),
    )


def _sample_indices_in_timestamp_window(
    store: CompiledProblemStore,
    *,
    start_timestamp_inclusive: int,
    end_timestamp_exclusive: int,
) -> np.ndarray:
    if end_timestamp_exclusive <= start_timestamp_inclusive:
        raise ValueError(
            "sample timestamp window end_timestamp_exclusive must be greater "
            "than start_timestamp_inclusive"
        )
    sample_timestamps = store.sample_timestamps(np.arange(store.n_samples, dtype=np.int64))
    mask = (
        (sample_timestamps >= start_timestamp_inclusive)
        & (sample_timestamps < end_timestamp_exclusive)
    )
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _sample_indices_in_block_window(
    store: CompiledProblemStore,
    *,
    start_block_inclusive: int,
    end_block_exclusive: int,
) -> np.ndarray:
    if end_block_exclusive <= start_block_inclusive:
        raise ValueError(
            "sample block window end_block_exclusive must be greater "
            "than start_block_inclusive"
        )
    all_indices = np.arange(store.n_samples, dtype=np.int64)
    sample_blocks = store.sample_block_numbers(all_indices)
    mask = (
        (sample_blocks >= start_block_inclusive)
        & (sample_blocks < end_block_exclusive)
    )
    selected = np.flatnonzero(mask).astype(np.int64, copy=False)
    expected_count = end_block_exclusive - start_block_inclusive
    if selected.size != expected_count:
        raise ValueError(
            "Evaluation corpus produced an incomplete block inference window: "
            f"expected {expected_count} samples, found {selected.size}"
        )
    return selected


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
    return transform_problem_store_features(store, scaler)


def prepare_training_dataset(
    blocks: pl.DataFrame,
    facts: TrainingDatasetPreparationFacts,
    context: TrainingDatasetPreparationContext,
    sequence: SequenceConfig,
) -> PreparedTrainingDataset:
    sorted_blocks = _prepare_blocks(blocks)
    capability_store = _build_selected_store(sorted_blocks, context=context)
    store_raw = capability_store.store
    raw_selected_sample_indices = _training_sample_indices(
        store_raw,
        training_cutoff_timestamp=facts.training_cutoff_timestamp,
    )
    raw_split_indices = _split_store_indices(
        raw_selected_sample_indices,
        facts=facts,
    )
    seq_len, median_dt_seconds = _compute_seq_len(
        _training_calibration_timestamps(store_raw, raw_split_indices.train),
        lookback_seconds=context.problem_contract.lookback_seconds,
        min_sequence_length=sequence.min_length,
        max_sequence_length=sequence.max_length,
    )
    history_seconds = context.problem_contract.feature_prerequisites.history_seconds
    warmup_rows = context.problem_contract.feature_prerequisites.warmup_rows
    store = _store_with_fixed_context(
        store_raw,
        context_length=seq_len,
        history_seconds=history_seconds,
        warmup_rows=warmup_rows,
    )
    selected_sample_indices = _training_sample_indices(
        store,
        training_cutoff_timestamp=facts.training_cutoff_timestamp,
    )
    split_indices = _split_store_indices(
        selected_sample_indices,
        facts=facts,
    )
    scaler = _train_scaler(
        store,
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
        sequence_runtime_metadata=sequence_runtime_metadata(
            sequence_length=int(seq_len),
            median_dt_seconds=float(median_dt_seconds),
            min_length=sequence.min_length,
            max_length=sequence.max_length,
        ),
        temporal_capability=capability_store.capability,
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    spec: CompiledInferenceDatasetPreparationRequest,
) -> PreparedInferenceDataset:
    combined_blocks = _prepare_blocks(pl.concat([history_blocks, evaluation_blocks]))
    runtime_metadata = spec.sequence_runtime_metadata
    feature_table = spec.feature_contract.build_table(combined_blocks)
    store = spec.problem_contract.build_delay_store(
        feature_table,
        spec.delay_seconds,
        capability=spec.temporal_capability,
    )
    store = _store_with_fixed_context(
        store,
        context_length=runtime_metadata.sequence_length,
        history_seconds=spec.problem_contract.feature_prerequisites.history_seconds,
        warmup_rows=spec.problem_contract.feature_prerequisites.warmup_rows,
    )
    if spec.sample_timestamp_window is not None:
        sample_indices = _sample_indices_in_timestamp_window(
            store,
            start_timestamp_inclusive=(
                spec.sample_timestamp_window.start_timestamp_inclusive
            ),
            end_timestamp_exclusive=(
                spec.sample_timestamp_window.end_timestamp_exclusive
            ),
        )
    elif spec.sample_block_window is not None:
        sample_indices = _sample_indices_in_block_window(
            store,
            start_block_inclusive=spec.sample_block_window.start_block_inclusive,
            end_block_exclusive=spec.sample_block_window.end_block_exclusive,
        )
    else:
        raise ValueError("inference preparation requires a sample window")
    if sample_indices.size == 0:
        raise ValueError("Evaluation corpus produced no valid inference examples")
    scaled_store = _scale_store(store, scaler=spec.scaler)
    return PreparedInferenceDataset(
        n_history_rows=history_blocks.height,
        n_evaluation_rows=evaluation_blocks.height,
        execution_policy=spec.problem_contract.execution_policy,
        store=scaled_store,
        samples=PreparedInferenceSampleSelection(
            action_space=spec.problem_contract.execution_policy.prepare_action_space(
                scaled_store,
                sample_indices,
            ),
        ),
    )

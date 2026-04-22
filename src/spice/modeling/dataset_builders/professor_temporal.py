"""Professor-aligned block-native dataset-preparation path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from ...core.errors import ConfigResolutionError
from ...temporal.contracts import ProblemRuntimeMetadata
from ...temporal.problem_store import (
    CompiledProblemStore,
    DatasetSplitIndices,
    filter_sample_indices_by_timestamp_window,
)
from ...temporal.scaling import transform_feature_matrix
from ..pipeline import (
    InferencePreparationSpec,
    PreparedInferenceDataset,
    PreparedTrainingDataset,
    TrainingSpec,
)
from .base import (
    CompiledDatasetBuilderContract,
    ProfessorTemporalDatasetBuilderConfig,
    builder_runtime_metadata,
    compiler_runtime_metadata_from_builder_payload,
)


def _prepare_blocks(blocks: pl.DataFrame) -> pl.DataFrame:
    if blocks.height == 0:
        raise ValueError("dataset builder received an empty block frame")
    return (
        blocks.sort("timestamp")
        .unique(subset=["block_number"], keep="first", maintain_order=True)
        .sort("block_number")
    )


def _split_row_bounds(
    n_rows: int,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    train_end = max(1, min(train_end, n_rows - 2))
    validation_end = max(train_end + 1, min(validation_end, n_rows - 1))
    return train_end, validation_end


def _compute_seq_len(
    timestamps: np.ndarray,
    *,
    lookback_seconds: int,
    min_sequence_length: int,
    max_sequence_length: int,
) -> tuple[int, float]:
    if timestamps.size < 2:
        raise ValueError("professor builder requires at least two timestamps")
    deltas = np.diff(timestamps.astype(np.float64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError("professor builder requires positive timestamp deltas")
    median_dt = float(np.median(positive_deltas))
    seq_len = int(round(lookback_seconds / max(median_dt, 1e-6)))
    seq_len = int(np.clip(seq_len, min_sequence_length, max_sequence_length))
    return seq_len, median_dt


def _build_selected_store(
    blocks: pl.DataFrame,
    *,
    spec: TrainingSpec,
) -> tuple[CompiledProblemStore, ProblemRuntimeMetadata]:
    feature_table = spec.feature_contract.build_table(_prepare_blocks(blocks))
    if feature_table.feature_prerequisites != spec.contract.feature_prerequisites:
        raise ValueError(
            "Resolved feature prerequisites do not match the current feature graph: "
            f"expected {spec.contract.feature_prerequisites.model_dump(mode='json')}, "
            f"got {feature_table.feature_prerequisites.model_dump(mode='json')}"
        )
    return spec.contract.build_capability_store(feature_table)


def _apply_sequence_length(
    store: CompiledProblemStore,
    *,
    seq_len: int,
    history_seconds: int,
    warmup_rows: int,
) -> CompiledProblemStore:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    context_start_rows = store.anchor_rows - seq_len + 1
    valid_anchor_mask = context_start_rows >= 0
    valid_anchor_mask &= context_start_rows >= warmup_rows
    if history_seconds > 0:
        valid_anchor_mask &= (
            store.timestamps[np.maximum(context_start_rows, 0)] - store.timestamps[0]
        ) >= history_seconds
    anchor_rows = store.anchor_rows[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("professor builder split produced no supervised samples")
    return replace(
        store,
        anchor_rows=anchor_rows,
        context_start_rows=context_start_rows[valid_anchor_mask].astype(np.int64, copy=False),
        candidate_start_rows=store.candidate_start_rows[valid_anchor_mask].astype(
            np.int64,
            copy=False,
        ),
        candidate_end_rows=store.candidate_end_rows[valid_anchor_mask].astype(
            np.int64,
            copy=False,
        ),
    )


def _train_scaler(
    store: CompiledProblemStore,
    *,
    spec: TrainingSpec,
    train_sample_indices: np.ndarray,
):
    return spec.input_normalization_contract.fit_scaler(
        store.feature_matrix,
        context_start_rows=store.context_start_rows,
        anchor_rows=store.anchor_rows,
        sample_indices=train_sample_indices.astype(np.int64, copy=False),
    )


def _split_store_indices(
    store: CompiledProblemStore,
    *,
    train_end_row: int,
    validation_end_row: int,
) -> DatasetSplitIndices:
    anchor_rows = store.anchor_rows.astype(np.int64, copy=False)
    train = np.flatnonzero(anchor_rows < train_end_row).astype(np.int64, copy=False)
    validation = np.flatnonzero(
        (anchor_rows >= train_end_row) & (anchor_rows < validation_end_row)
    ).astype(np.int64, copy=False)
    test = np.flatnonzero(anchor_rows >= validation_end_row).astype(np.int64, copy=False)
    if train.size == 0 or validation.size == 0 or test.size == 0:
        raise ValueError("professor builder produced an empty train/validation/test split")
    return DatasetSplitIndices(train=train, validation=validation, test=test)


def _scale_store(
    store: CompiledProblemStore,
    *,
    scaler,
) -> CompiledProblemStore:
    return replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )


def prepare_training_dataset(blocks: pl.DataFrame, spec: TrainingSpec) -> PreparedTrainingDataset:
    if not isinstance(spec.dataset_builder, ProfessorTemporalDatasetBuilderConfig):
        raise TypeError("professor builder requires ProfessorTemporalDatasetBuilderConfig")
    sorted_blocks = _prepare_blocks(blocks)
    if sorted_blocks.height < spec.problem.sample_count:
        raise ValueError(
            "History dataset is too short for the requested sample count; "
            f"need at least {spec.problem.sample_count} rows, got {sorted_blocks.height}"
        )
    selected_blocks = sorted_blocks.tail(spec.problem.sample_count)
    train_end, validation_end = _split_row_bounds(
        selected_blocks.height,
        train_fraction=spec.split.train_fraction,
        validation_fraction=spec.split.validation_fraction,
    )
    seq_len, median_dt_seconds = _compute_seq_len(
        selected_blocks.slice(0, train_end)["timestamp"]
        .cast(pl.Int64)
        .to_numpy()
        .astype(np.int64, copy=False),
        lookback_seconds=spec.problem.lookback_seconds,
        min_sequence_length=spec.dataset_builder.min_sequence_length,
        max_sequence_length=spec.dataset_builder.max_sequence_length,
    )
    history_seconds = spec.contract.feature_prerequisites.history_seconds
    warmup_rows = spec.contract.feature_prerequisites.warmup_rows
    store_raw, compiler_runtime_metadata = _build_selected_store(selected_blocks, spec=spec)
    store = _apply_sequence_length(
        store_raw,
        seq_len=seq_len,
        history_seconds=history_seconds,
        warmup_rows=warmup_rows,
    )
    split_indices = _split_store_indices(
        store,
        train_end_row=train_end,
        validation_end_row=validation_end,
    )
    scaler = _train_scaler(
        store,
        spec=spec,
        train_sample_indices=split_indices.train,
    )
    scaled_store = _scale_store(store, scaler=scaler)
    return PreparedTrainingDataset(
        n_rows_available=sorted_blocks.height,
        n_rows_used=scaled_store.n_rows,
        sample_count=scaled_store.n_samples,
        feature=spec.feature_contract.semantics,
        realization_policy=spec.contract.realization_policy,
        store=scaled_store,
        split_indices=split_indices,
        scaler=scaler,
        builder_runtime_metadata=builder_runtime_metadata(
            compiler_runtime_metadata=compiler_runtime_metadata,
            extra={
                "seq_len": int(seq_len),
                "median_dt_seconds": float(median_dt_seconds),
                "min_sequence_length": spec.dataset_builder.min_sequence_length,
                "max_sequence_length": spec.dataset_builder.max_sequence_length,
                "split_strategy": "global_feature_table",
            },
        ),
    )


def prepare_inference_dataset(
    history_blocks: pl.DataFrame,
    evaluation_blocks: pl.DataFrame,
    spec: InferencePreparationSpec,
) -> PreparedInferenceDataset:
    combined_blocks = _prepare_blocks(pl.concat([history_blocks, evaluation_blocks]))
    compiler_runtime_metadata = compiler_runtime_metadata_from_builder_payload(
        spec.builder_runtime_metadata,
        compiler_id=spec.contract.compiler_id,
    )
    seq_len = _builder_int(spec.builder_runtime_metadata, "seq_len")
    feature_table = spec.feature_contract.build_table(combined_blocks)
    store = spec.contract.build_delay_store(
        feature_table,
        spec.delay_seconds,
        compiler_runtime_metadata=compiler_runtime_metadata,
        max_candidate_slots=spec.max_candidate_slots,
    )
    store = _apply_sequence_length(
        store,
        seq_len=seq_len,
        history_seconds=spec.contract.feature_prerequisites.history_seconds,
        warmup_rows=spec.contract.feature_prerequisites.warmup_rows,
    )
    sample_indices = filter_sample_indices_by_timestamp_window(
        store,
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
        feature=spec.feature_contract.semantics,
        realization_policy=spec.contract.realization_policy,
        store=scaled_store,
        sample_indices=sample_indices,
    )


def _builder_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigResolutionError(f"builder runtime metadata field must be an integer: {key}")
    return int(value)


def compile_dataset_builder(
    config: ProfessorTemporalDatasetBuilderConfig,
) -> CompiledDatasetBuilderContract:
    if config.id != "professor_temporal":
        raise ConfigResolutionError("dataset_builder.id must be professor_temporal")
    return CompiledDatasetBuilderContract(
        dataset_builder_id="professor_temporal",
        prepare_training_fn=prepare_training_dataset,
        prepare_inference_fn=prepare_inference_dataset,
    )

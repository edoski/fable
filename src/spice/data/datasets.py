"""Dataset geometry and array-backed temporal stores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.config import SplitConfig
from .features import FeatureTable, feature_warmup_blocks

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(slots=True)
class DatasetGeometry:
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    context_block_count: int

    def required_block_count(self, sample_count: int) -> int:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return self.context_block_count + sample_count + self.action_count


@dataclass(slots=True)
class DatasetSplitIndices:
    train: IntVector
    validation: IntVector
    test: IntVector


@dataclass(slots=True)
class TemporalDatasetStore:
    feature_matrix: FloatMatrix
    block_numbers: IntVector
    timestamps: IntVector
    sample_row_indices: IntVector
    class_labels: IntVector
    action_log_fees: FloatMatrix
    target_log_fee: FloatVector
    next_block_log_fee: FloatVector
    optimal_log_fee: FloatVector

    @property
    def n_rows(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.feature_matrix.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.sample_row_indices.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.action_log_fees.shape[1])


def lookback_steps_for_seconds(lookback_seconds: int, block_time_seconds: float) -> int:
    if block_time_seconds <= 0:
        raise ValueError("block_time_seconds must be positive")
    return max(1, round(lookback_seconds / block_time_seconds))


def max_extra_wait_steps_for_delay(max_delay_seconds: int, block_time_seconds: float) -> int:
    if block_time_seconds <= 0:
        raise ValueError("block_time_seconds must be positive")
    return max(1, math.floor(max_delay_seconds / block_time_seconds))


def action_count_for_delay(max_delay_seconds: int, block_time_seconds: float) -> int:
    return max_extra_wait_steps_for_delay(max_delay_seconds, block_time_seconds) + 1


def derive_dataset_geometry(
    *,
    lookback_seconds: int,
    max_delay_seconds: int,
    block_time_seconds: float,
) -> DatasetGeometry:
    lookback_steps = lookback_steps_for_seconds(lookback_seconds, block_time_seconds)
    max_extra_wait_steps = max_extra_wait_steps_for_delay(max_delay_seconds, block_time_seconds)
    action_count = action_count_for_delay(max_delay_seconds, block_time_seconds)
    warmup_blocks = feature_warmup_blocks()
    return DatasetGeometry(
        lookback_steps=lookback_steps,
        max_extra_wait_steps=max_extra_wait_steps,
        action_count=action_count,
        context_block_count=warmup_blocks + lookback_steps - 1,
    )


def trim_history_for_sample_count(
    n_blocks: int,
    *,
    sample_count: int,
    geometry: DatasetGeometry,
) -> slice:
    required_blocks = geometry.required_block_count(sample_count)
    if n_blocks < required_blocks:
        raise ValueError(
            "History dataset is too short for the requested sample count; "
            f"need at least {required_blocks} blocks, got {n_blocks}"
        )
    return slice(n_blocks - required_blocks, n_blocks)


def history_context_slice(n_blocks: int, *, geometry: DatasetGeometry) -> slice:
    if n_blocks < geometry.context_block_count:
        raise ValueError(
            "History dataset is too short to provide evaluation context; "
            f"need at least {geometry.context_block_count} blocks, got {n_blocks}"
        )
    return slice(n_blocks - geometry.context_block_count, n_blocks)


def build_temporal_store(
    feature_table: FeatureTable,
    *,
    lookback_steps: int,
    action_count: int,
) -> TemporalDatasetStore:
    if lookback_steps <= 0:
        raise ValueError("lookback_steps must be positive")
    if action_count <= 0:
        raise ValueError("action_count must be positive")

    max_sample_row = len(feature_table.block_numbers) - action_count
    if max_sample_row <= lookback_steps - 1:
        raise ValueError("Feature table is too short to produce any supervised samples")

    sample_row_indices = np.arange(lookback_steps - 1, max_sample_row, dtype=np.int64)
    future_windows = np.lib.stride_tricks.sliding_window_view(
        feature_table.log_base_fees[1:],
        window_shape=action_count,
    )
    action_log_fees = future_windows[sample_row_indices].astype(np.float32, copy=False)
    class_labels = np.argmin(action_log_fees, axis=1).astype(np.int64, copy=False)
    row_selector = np.arange(class_labels.shape[0], dtype=np.int64)
    target_log_fee = action_log_fees[row_selector, class_labels].astype(np.float32, copy=False)
    next_block_log_fee = action_log_fees[:, 0].astype(np.float32, copy=False)
    optimal_log_fee = action_log_fees.min(axis=1).astype(np.float32, copy=False)

    return TemporalDatasetStore(
        feature_matrix=feature_table.feature_matrix,
        block_numbers=feature_table.block_numbers,
        timestamps=feature_table.timestamps,
        sample_row_indices=sample_row_indices,
        class_labels=class_labels,
        action_log_fees=action_log_fees,
        target_log_fee=target_log_fee,
        next_block_log_fee=next_block_log_fee,
        optimal_log_fee=optimal_log_fee,
    )


def filter_sample_indices_by_timestamp_window(
    store: TemporalDatasetStore,
    *,
    start_timestamp: int,
    end_timestamp: int,
) -> IntVector:
    sample_timestamps = store.timestamps[store.sample_row_indices]
    mask = (sample_timestamps >= start_timestamp) & (sample_timestamps < end_timestamp)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def chronological_split_indices(
    n_samples: int,
    split_config: SplitConfig,
) -> DatasetSplitIndices:
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

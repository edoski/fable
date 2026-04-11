"""History and evaluation window planning for acquisition."""

from __future__ import annotations

from math import ceil

from ..core.config import ExperimentConfig
from ..data.datasets import derive_dataset_geometry
from .cryo import TimestampRange, history_range_for_required_blocks
from .metadata import DatasetMetadata
from .raw_validation import RawPullValidationReport


def required_history_block_count(config: ExperimentConfig) -> int:
    geometry = derive_dataset_geometry(
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        block_time_seconds=config.chain.block_time_seconds,
    )
    return geometry.required_block_count(config.dataset.sampling.effective_history_anchor_count)


def initial_history_range(
    config: ExperimentConfig,
    *,
    required_history_blocks: int,
) -> TimestampRange:
    return history_range_for_required_blocks(
        config.chain,
        config.acquisition,
        required_history_blocks=required_history_blocks,
        window_start_timestamp=config.dataset.window.start_timestamp,
    )


def expanded_history_range(
    current: TimestampRange,
    validation: RawPullValidationReport,
    *,
    config: ExperimentConfig,
    required_history_blocks: int,
) -> TimestampRange:
    missing_blocks = required_history_blocks - validation.row_count
    if missing_blocks <= 0:
        return current
    if (
        validation.first_timestamp is not None
        and validation.last_timestamp is not None
        and validation.row_count > 1
    ):
        seconds_per_block = max(
            config.chain.block_time_seconds,
            (validation.last_timestamp - validation.first_timestamp) / (validation.row_count - 1),
        )
    else:
        seconds_per_block = config.chain.block_time_seconds
    additional_blocks = missing_blocks + config.acquisition.chunk_size
    return TimestampRange(
        start=current.start - ceil(additional_blocks * seconds_per_block),
        end=current.end,
    )


def history_range_from_metadata(metadata: DatasetMetadata) -> TimestampRange:
    history = metadata.windows.history
    return TimestampRange(
        start=history.start_timestamp,
        end=history.end_timestamp,
    )

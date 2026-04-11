"""Acquisition planning helpers."""

from __future__ import annotations

from ..core.config import ExperimentConfig
from ..data.datasets import derive_dataset_geometry
from .metadata import DatasetMetadata
from .rpc import TimestampRange


def required_history_block_count(config: ExperimentConfig) -> int:
    geometry = derive_dataset_geometry(
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        block_time_seconds=config.chain.block_time_seconds,
    )
    return geometry.required_block_count(config.dataset.sampling.effective_history_anchor_count)


def history_range_from_metadata(metadata: DatasetMetadata) -> TimestampRange:
    history = metadata.windows.history
    return TimestampRange(
        start=history.start_timestamp,
        end=history.end_timestamp,
    )

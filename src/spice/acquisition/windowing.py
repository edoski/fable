"""Acquisition planning helpers."""

from __future__ import annotations

from ..core.config import ExperimentConfig
from ..data.datasets import derive_dataset_geometry


def required_history_block_count(config: ExperimentConfig) -> int:
    geometry = derive_dataset_geometry(
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        block_time_seconds=config.chain.block_time_seconds,
    )
    return geometry.required_block_count(config.effective_history_sample_budget)

"""Pure dataset geometry and planning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class DatasetGeometry:
    lookback_steps: int
    max_extra_wait_steps: int
    action_count: int
    history_context_blocks: int

    def required_block_count(self, sample_count: int) -> int:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return self.history_context_blocks + sample_count + self.action_count


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


def minimum_history_context_blocks(
    *,
    lookback_seconds: int,
    block_time_seconds: float,
    feature_warmup_blocks: int,
) -> int:
    return (
        feature_warmup_blocks
        + lookback_steps_for_seconds(lookback_seconds, block_time_seconds)
        - 1
    )


def derive_dataset_geometry(
    *,
    lookback_seconds: int,
    max_delay_seconds: int,
    block_time_seconds: float,
    history_context_blocks: int,
) -> DatasetGeometry:
    lookback_steps = lookback_steps_for_seconds(lookback_seconds, block_time_seconds)
    max_extra_wait_steps = max_extra_wait_steps_for_delay(max_delay_seconds, block_time_seconds)
    action_count = action_count_for_delay(max_delay_seconds, block_time_seconds)
    minimum_context = lookback_steps - 1
    if history_context_blocks < minimum_context:
        raise ValueError(
            "history context is too small for the configured lookback; "
            f"need at least {minimum_context}, got {history_context_blocks}"
        )
    return DatasetGeometry(
        lookback_steps=lookback_steps,
        max_extra_wait_steps=max_extra_wait_steps,
        action_count=action_count,
        history_context_blocks=history_context_blocks,
    )

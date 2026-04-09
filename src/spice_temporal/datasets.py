"""Dataset construction logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

from spice_temporal.config import SplitConfig
from spice_temporal.records import FeatureRow, SupervisedExample


@dataclass(slots=True)
class DatasetSplit:
    train: list[SupervisedExample]
    validation: list[SupervisedExample]
    test: list[SupervisedExample]


def estimate_horizon_blocks(window_seconds: int, average_interblock_seconds: float) -> int:
    if average_interblock_seconds <= 0:
        raise ValueError("average_interblock_seconds must be positive")
    return max(1, math.floor(window_seconds / average_interblock_seconds))


def average_interblock_seconds(timestamps: list[int]) -> float:
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to estimate inter-block time")
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:], strict=False)]
    return sum(deltas) / len(deltas)


def earliest_min_offset(values: list[float]) -> int:
    if not values:
        raise ValueError("Cannot choose a minimum offset from an empty list")
    min_value = min(values)
    for index, value in enumerate(values):
        if value == min_value:
            return index
    raise RuntimeError("Unreachable")


def build_supervised_examples(
    feature_rows: list[FeatureRow],
    *,
    lookback_steps: int,
    horizon_blocks: int,
) -> list[SupervisedExample]:
    if lookback_steps <= 0:
        raise ValueError("lookback_steps must be positive")
    if horizon_blocks <= 0:
        raise ValueError("horizon_blocks must be positive")

    examples: list[SupervisedExample] = []
    max_anchor = len(feature_rows) - horizon_blocks
    for anchor_index in range(lookback_steps - 1, max_anchor):
        sequence_start = anchor_index - lookback_steps + 1
        sequence = feature_rows[sequence_start : anchor_index + 1]
        future = feature_rows[anchor_index + 1 : anchor_index + 1 + horizon_blocks]
        future_log_fees = [row.log_base_fee for row in future]
        min_offset = earliest_min_offset(future_log_fees)
        examples.append(
            SupervisedExample(
                anchor_block_number=feature_rows[anchor_index].block_number,
                anchor_timestamp=feature_rows[anchor_index].timestamp,
                inputs=[row.features for row in sequence],
                class_label=min_offset,
                target_log_fee=future_log_fees[min_offset],
                future_log_fees=future_log_fees,
                next_block_log_fee=future_log_fees[0],
                optimal_log_fee=min(future_log_fees),
            )
        )
    return examples


def chronological_split(
    examples: list[SupervisedExample],
    split_config: SplitConfig,
) -> DatasetSplit:
    total = len(examples)
    if total < 3:
        raise ValueError("Need at least three examples to create train/validation/test splits")

    train_end = int(total * split_config.train_fraction)
    validation_end = train_end + int(total * split_config.validation_fraction)
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))
    return DatasetSplit(
        train=examples[:train_end],
        validation=examples[train_end:validation_end],
        test=examples[validation_end:],
    )

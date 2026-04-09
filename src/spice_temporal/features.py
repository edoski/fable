"""Feature engineering for temporal SPICE baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spice_temporal.records import BlockRecord

ROLLING_WINDOWS = (10, 50, 200)
FEATURE_NAMES = (
    "log_base_fee",
    "gas_utilization",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "elapsed_blocks",
    "trend_slope_200",
) + tuple(
    feature_name
    for window in ROLLING_WINDOWS
    for feature_name in (
        f"rolling_mean_log_base_fee_{window}",
        f"rolling_std_log_base_fee_{window}",
        f"rolling_mean_gas_utilization_{window}",
        f"rolling_std_gas_utilization_{window}",
    )
)
N_FEATURES = len(FEATURE_NAMES)

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(slots=True)
class FeatureTable:
    block_numbers: IntVector
    timestamps: IntVector
    feature_matrix: FloatMatrix
    log_base_fees: FloatVector


def feature_warmup_blocks() -> int:
    return max(ROLLING_WINDOWS) - 1


def cyclical_encode(value: int, period: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def safe_log(value: float) -> float:
    return math.log(max(value, 1.0))


def rolling_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def rolling_std(values: Sequence[float]) -> float:
    if len(values) == 1:
        return 0.0
    mean = rolling_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def ols_slope(values: Sequence[float]) -> float:
    n_values = len(values)
    if n_values < 2:
        return 0.0
    x_mean = (n_values - 1) / 2
    y_mean = rolling_mean(values)
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        dx = index - x_mean
        numerator += dx * (value - y_mean)
        denominator += dx * dx
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def build_feature_table(blocks: Sequence[BlockRecord]) -> FeatureTable:
    if not blocks:
        return FeatureTable(
            block_numbers=np.empty(0, dtype=np.int64),
            timestamps=np.empty(0, dtype=np.int64),
            feature_matrix=np.empty((0, N_FEATURES), dtype=np.float32),
            log_base_fees=np.empty(0, dtype=np.float32),
        )

    sorted_blocks = sorted(blocks, key=lambda item: item.block_number)
    log_base_fees_all = [safe_log(block.base_fee_per_gas) for block in sorted_blocks]
    gas_utilizations = [block.gas_utilization for block in sorted_blocks]
    min_index = feature_warmup_blocks()
    n_rows = len(sorted_blocks) - min_index
    feature_matrix = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    block_numbers = np.empty(n_rows, dtype=np.int64)
    timestamps = np.empty(n_rows, dtype=np.int64)
    log_base_fees = np.empty(n_rows, dtype=np.float32)

    for row_index, block_index in enumerate(range(min_index, len(sorted_blocks))):
        block = sorted_blocks[block_index]
        hour = (block.timestamp // 3600) % 24
        weekday = (block.timestamp // 86_400 + 4) % 7
        hour_sin, hour_cos = cyclical_encode(hour, 24)
        weekday_sin, weekday_cos = cyclical_encode(weekday, 7)

        feature_vector = [
            log_base_fees_all[block_index],
            gas_utilizations[block_index],
            hour_sin,
            hour_cos,
            weekday_sin,
            weekday_cos,
            float(block_index),
            ols_slope(log_base_fees_all[block_index - 199 : block_index + 1]),
        ]

        for window in ROLLING_WINDOWS:
            fee_window = log_base_fees_all[block_index - window + 1 : block_index + 1]
            gas_window = gas_utilizations[block_index - window + 1 : block_index + 1]
            feature_vector.extend(
                [
                    rolling_mean(fee_window),
                    rolling_std(fee_window),
                    rolling_mean(gas_window),
                    rolling_std(gas_window),
                ]
            )

        block_numbers[row_index] = block.block_number
        timestamps[row_index] = block.timestamp
        feature_matrix[row_index] = np.asarray(feature_vector, dtype=np.float32)
        log_base_fees[row_index] = np.float32(log_base_fees_all[block_index])

    return FeatureTable(
        block_numbers=block_numbers,
        timestamps=timestamps,
        feature_matrix=feature_matrix,
        log_base_fees=log_base_fees,
    )

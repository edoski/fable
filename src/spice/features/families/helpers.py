"""Shared helpers for concrete feature families."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..core import CanonicalBlockSeries

FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


def log_base_fee_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del blocks, resolved_dependencies
    return series.log_base_fees.astype(np.float64, copy=False)


def gas_utilization_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del series, resolved_dependencies
    return gas_utilization(blocks)


def hour_sin_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del blocks, resolved_dependencies
    return hour_sin(series.timestamps)


def hour_cos_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del blocks, resolved_dependencies
    return hour_cos(series.timestamps)


def weekday_sin_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del blocks, resolved_dependencies
    return weekday_sin(series.timestamps)


def weekday_cos_feature(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: object,
) -> FloatVector:
    del blocks, resolved_dependencies
    return weekday_cos(series.timestamps)


def build_canonical_series(blocks: pl.DataFrame) -> CanonicalBlockSeries:
    return CanonicalBlockSeries(
        block_numbers=blocks["block_number"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
        timestamps=blocks["timestamp"].cast(pl.Int64).to_numpy().astype(np.int64, copy=False),
        log_base_fees=(
            blocks["base_fee_per_gas"]
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
            .log()
            .to_numpy()
            .astype(np.float32, copy=False)
        ),
    )


def gas_utilization(blocks: pl.DataFrame) -> FloatVector:
    return (
        blocks["gas_used"].cast(pl.Float64) / blocks["gas_limit"].cast(pl.Float64)
    ).to_numpy().astype(np.float64, copy=False)


def hour_sin(timestamps: IntVector) -> FloatVector:
    hours = (timestamps // 3600) % 24
    return np.sin(2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0)


def hour_cos(timestamps: IntVector) -> FloatVector:
    hours = (timestamps // 3600) % 24
    return np.cos(2.0 * math.pi * hours.astype(np.float64, copy=False) / 24.0)


def weekday_sin(timestamps: IntVector) -> FloatVector:
    weekdays = ((timestamps // 86_400) + 4) % 7
    return np.sin(2.0 * math.pi * weekdays.astype(np.float64, copy=False) / 7.0)


def weekday_cos(timestamps: IntVector) -> FloatVector:
    weekdays = ((timestamps // 86_400) + 4) % 7
    return np.cos(2.0 * math.pi * weekdays.astype(np.float64, copy=False) / 7.0)


def block_rolling_stat(values: FloatVector, *, window: int, stat: str) -> FloatVector:
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    series = pl.Series(values)
    if stat == "mean":
        result = series.rolling_mean(window_size=window, min_samples=window)
    elif stat == "std":
        result = series.rolling_std(window_size=window, min_samples=window, ddof=0)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported rolling stat: {stat}")
    return result.to_numpy().astype(np.float64, copy=False)


def block_trend_slope(values: FloatVector, *, window: int) -> FloatVector:
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    centered_x = np.arange(window, dtype=np.float64) - (window - 1) / 2
    denominator = np.square(centered_x).sum()
    window_means = windows.mean(axis=1, keepdims=True)
    result[window - 1 :] = ((windows - window_means) * centered_x).sum(axis=1) / denominator
    return result


def time_window_bounds(timestamps: IntVector, *, window_seconds: int) -> IntVector:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    starts = np.searchsorted(timestamps, timestamps - window_seconds, side="left")
    valid = (timestamps - timestamps[0]) >= window_seconds
    return np.where(valid, starts, -1).astype(np.int64, copy=False)


def time_rolling_mean(values: FloatVector, starts: IntVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size == 0:
        return result
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    for index, start in enumerate(starts):
        if start < 0:
            continue
        count = index - start + 1
        result[index] = (cumsum[index + 1] - cumsum[start]) / count
    return result


def time_rolling_std(values: FloatVector, starts: IntVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size == 0:
        return result
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    square_cumsum = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
    for index, start in enumerate(starts):
        if start < 0:
            continue
        count = index - start + 1
        mean = (cumsum[index + 1] - cumsum[start]) / count
        second_moment = (square_cumsum[index + 1] - square_cumsum[start]) / count
        variance = max(0.0, second_moment - mean * mean)
        result[index] = variance**0.5
    return result


def time_trend_slope(
    values: FloatVector,
    timestamps: IntVector,
    *,
    window_seconds: int,
) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    starts = time_window_bounds(timestamps, window_seconds=window_seconds)
    for index, start in enumerate(starts):
        if start < 0:
            continue
        window_values = values[start : index + 1]
        window_times = timestamps[start : index + 1].astype(np.float64, copy=False)
        centered_x = window_times - window_times.mean()
        denominator = np.square(centered_x).sum()
        if denominator <= 0.0:
            result[index] = 0.0
            continue
        centered_y = window_values - window_values.mean()
        result[index] = float((centered_x * centered_y).sum() / denominator)
    return result

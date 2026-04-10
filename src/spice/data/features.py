"""Feature engineering for temporal SPICE baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

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


def _trend_slopes(log_base_fees: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    windows = np.lib.stride_tricks.sliding_window_view(log_base_fees, window_shape=window)
    centered_x = np.arange(window, dtype=np.float64) - (window - 1) / 2
    denominator = np.square(centered_x).sum()
    window_means = windows.mean(axis=1, keepdims=True)
    return ((windows - window_means) * centered_x).sum(axis=1) / denominator


def build_feature_table(blocks: pl.DataFrame) -> FeatureTable:
    if blocks.height == 0:
        return FeatureTable(
            block_numbers=np.empty(0, dtype=np.int64),
            timestamps=np.empty(0, dtype=np.int64),
            feature_matrix=np.empty((0, N_FEATURES), dtype=np.float32),
            log_base_fees=np.empty(0, dtype=np.float32),
        )

    frame = (
        blocks.sort("block_number")
        .with_row_index("elapsed_blocks")
        .with_columns(
            [
                pl.col("base_fee_per_gas")
                .cast(pl.Float64)
                .clip(lower_bound=1.0)
                .log()
                .alias("log_base_fee"),
                (pl.col("gas_used").cast(pl.Float64) / pl.col("gas_limit").cast(pl.Float64)).alias(
                    "gas_utilization"
                ),
                (
                    2.0
                    * math.pi
                    * ((pl.col("timestamp") // 3600) % 24).cast(pl.Float64)
                    / 24.0
                ).sin().alias("hour_sin"),
                (
                    2.0
                    * math.pi
                    * ((pl.col("timestamp") // 3600) % 24).cast(pl.Float64)
                    / 24.0
                ).cos().alias("hour_cos"),
                (
                    2.0
                    * math.pi
                    * (((pl.col("timestamp") // 86_400) + 4) % 7).cast(pl.Float64)
                    / 7.0
                ).sin().alias("weekday_sin"),
                (
                    2.0
                    * math.pi
                    * (((pl.col("timestamp") // 86_400) + 4) % 7).cast(pl.Float64)
                    / 7.0
                ).cos().alias("weekday_cos"),
                pl.col("elapsed_blocks").cast(pl.Float64),
            ]
        )
    )

    for window in ROLLING_WINDOWS:
        frame = frame.with_columns(
            [
                pl.col("log_base_fee")
                .rolling_mean(window_size=window, min_samples=window)
                .alias(f"rolling_mean_log_base_fee_{window}"),
                pl.col("log_base_fee")
                .rolling_std(window_size=window, min_samples=window, ddof=0)
                .alias(f"rolling_std_log_base_fee_{window}"),
                pl.col("gas_utilization")
                .rolling_mean(window_size=window, min_samples=window)
                .alias(f"rolling_mean_gas_utilization_{window}"),
                pl.col("gas_utilization")
                .rolling_std(window_size=window, min_samples=window, ddof=0)
                .alias(f"rolling_std_gas_utilization_{window}"),
            ]
        )

    warmup = feature_warmup_blocks()
    log_base_fees_all = frame["log_base_fee"].to_numpy().astype(np.float64, copy=False)
    slopes = _trend_slopes(log_base_fees_all, 200)
    trimmed = frame.slice(warmup).with_columns(
        pl.Series(name="trend_slope_200", values=slopes)
    )

    feature_matrix = (
        trimmed.select(FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    )
    block_numbers = trimmed["block_number"].cast(pl.Int64).to_numpy()
    timestamps = trimmed["timestamp"].cast(pl.Int64).to_numpy()
    log_base_fees = trimmed["log_base_fee"].cast(pl.Float32).to_numpy()
    return FeatureTable(
        block_numbers=block_numbers.astype(np.int64, copy=False),
        timestamps=timestamps.astype(np.int64, copy=False),
        feature_matrix=feature_matrix,
        log_base_fees=log_base_fees.astype(np.float32, copy=False),
    )

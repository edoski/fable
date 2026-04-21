"""Block-native feature family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ..core import CanonicalBlockSeries
from . import helpers
from .base import FeatureDefinition, FeatureFamily, FeatureFamilyConfig

FloatVector = helpers.FloatVector


class BlockNativeFeatureFamilyConfig(FeatureFamilyConfig):
    id: str = "block_native"


def _elapsed_blocks(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks, resolved_dependencies
    if series.block_numbers.size == 0:
        return np.empty(0, dtype=np.float64)
    return series.block_numbers.astype(np.float64, copy=False) - float(series.block_numbers[0])


def _log1p_column(
    blocks: pl.DataFrame,
    column: str,
) -> FloatVector:
    return np.log1p(
        blocks[column].cast(pl.Float64).to_numpy().astype(np.float64, copy=False)
    )


def _seconds_since_previous_block(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks, resolved_dependencies
    if series.timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.empty(series.timestamps.shape[0], dtype=np.float64)
    result[0] = np.nan
    if series.timestamps.size == 1:
        return result
    deltas = np.diff(series.timestamps.astype(np.float64, copy=False))
    median_delta = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
    result[0] = median_delta
    result[1:] = deltas
    return result


def _delta(
    values: FloatVector,
) -> FloatVector:
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.empty(values.shape[0], dtype=np.float64)
    result[0] = np.nan
    if values.shape[0] > 1:
        result[1:] = np.diff(values)
    return result


def _shift(
    values: FloatVector,
    *,
    lag: int,
) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if values.size <= lag:
        return result
    result[lag:] = values[:-lag]
    return result


def _rolling_stat(
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window: int,
    stat: str,
    ddof: int = 0,
) -> FloatVector:
    return helpers.block_rolling_stat(
        resolved_dependencies[dependency_name],
        window=window,
        stat=stat,
        ddof=ddof,
    )


def _trend_slope(
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window: int,
) -> FloatVector:
    return helpers.block_trend_slope(resolved_dependencies[dependency_name], window=window)


def _log_base_fee_per_gas(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del series, resolved_dependencies
    return _log1p_column(blocks, "base_fee_per_gas")


def _log_gas_used(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del series, resolved_dependencies
    return _log1p_column(blocks, "gas_used")


def _log_gas_limit(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del series, resolved_dependencies
    return _log1p_column(blocks, "gas_limit")


def _feature_definitions() -> dict[str, FeatureDefinition]:
    features: dict[str, FeatureDefinition] = {
        "log_base_fee": FeatureDefinition((), 0, 0, helpers.log_base_fee_feature),
        "gas_utilization": FeatureDefinition(
            (),
            0,
            0,
            helpers.gas_utilization_feature,
        ),
        "hour_sin": FeatureDefinition((), 0, 0, helpers.hour_sin_feature),
        "hour_cos": FeatureDefinition((), 0, 0, helpers.hour_cos_feature),
        "weekday_sin": FeatureDefinition((), 0, 0, helpers.weekday_sin_feature),
        "weekday_cos": FeatureDefinition((), 0, 0, helpers.weekday_cos_feature),
        "elapsed_blocks": FeatureDefinition((), 0, 0, _elapsed_blocks),
        "log_base_fee_per_gas": FeatureDefinition(
            (),
            0,
            0,
            _log_base_fee_per_gas,
        ),
        "log_gas_used": FeatureDefinition((), 0, 0, _log_gas_used),
        "log_gas_limit": FeatureDefinition((), 0, 0, _log_gas_limit),
        "seconds_since_previous_block": FeatureDefinition(
            (),
            0,
            0,
            _seconds_since_previous_block,
        ),
        "dlog_base_fee": FeatureDefinition(
            ("log_base_fee_per_gas",),
            0,
            1,
            lambda blocks, series, resolved_dependencies: _delta(
                resolved_dependencies["log_base_fee_per_gas"]
            ),
        ),
        "trend_slope_200": FeatureDefinition(
            ("log_base_fee",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _trend_slope(
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=200,
            ),
        ),
        "trend_slope_log_base_fee_per_gas_200": FeatureDefinition(
            ("log_base_fee_per_gas",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _trend_slope(
                resolved_dependencies,
                dependency_name="log_base_fee_per_gas",
                window=200,
            ),
        ),
    }
    rolling_specs = (
        ("rolling_mean_log_base_fee", "log_base_fee", "mean", 0),
        ("rolling_std_log_base_fee", "log_base_fee", "std", 0),
        ("rolling_mean_gas_utilization", "gas_utilization", "mean", 0),
        ("rolling_std_gas_utilization", "gas_utilization", "std", 0),
        ("rolling_mean_log_base_fee_per_gas", "log_base_fee_per_gas", "mean", 0),
        ("rolling_std_log_base_fee_per_gas", "log_base_fee_per_gas", "std", 1),
        ("rolling_min_log_base_fee_per_gas", "log_base_fee_per_gas", "min", 0),
    )
    for window in (10, 50, 200):
        for prefix, dependency_name, stat, ddof in rolling_specs:
            name = f"{prefix}_{window}"
            features[name] = FeatureDefinition(
                (dependency_name,),
                0,
                window - 1,
                lambda blocks,
                series,
                resolved_dependencies,
                dependency_name=dependency_name,
                window=window,
                stat=stat,
                ddof=ddof: _rolling_stat(
                    resolved_dependencies,
                    dependency_name=dependency_name,
                    window=window,
                    stat=stat,
                    ddof=ddof,
                ),
            )
    for lag in range(1, 7):
        features[f"gas_utilization_lag{lag}"] = FeatureDefinition(
            ("gas_utilization",),
            0,
            lag,
            lambda blocks, series, resolved_dependencies, lag=lag: _shift(
                resolved_dependencies["gas_utilization"],
                lag=lag,
            ),
        )
        features[f"dlog_base_fee_lag{lag}"] = FeatureDefinition(
            ("dlog_base_fee",),
            0,
            lag + 1,
            lambda blocks, series, resolved_dependencies, lag=lag: _shift(
                resolved_dependencies["dlog_base_fee"],
                lag=lag,
            ),
        )
    return features


BLOCK_NATIVE_FAMILY = FeatureFamily(
    features=_feature_definitions(),
    fingerprint_sources=(Path(__file__).resolve(), Path(helpers.__file__).resolve()),
    build_series=helpers.build_canonical_series,
)

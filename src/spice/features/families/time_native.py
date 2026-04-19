"""Time-native feature family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ..core import CanonicalBlockSeries
from . import helpers
from .base import FeatureDefinition, FeatureFamilyConfig, FeatureFamilySpec

FloatVector = helpers.FloatVector


class TimeNativeFeatureFamilyConfig(FeatureFamilyConfig):
    id: str = "time_native"


def _seconds_since_previous_block(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks, resolved_dependencies
    if series.timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    deltas = np.diff(
        series.timestamps,
        prepend=series.timestamps[:1],
    ).astype(np.float64, copy=False)
    deltas[0] = 0.0
    return deltas


def _elapsed_seconds(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks, resolved_dependencies
    if series.timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    return series.timestamps.astype(np.float64, copy=False) - float(series.timestamps[0])


def _time_rolling_mean(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window_seconds: int,
) -> FloatVector:
    del blocks
    starts = helpers.time_window_bounds(series.timestamps, window_seconds=window_seconds)
    return helpers.time_rolling_mean(resolved_dependencies[dependency_name], starts)


def _time_rolling_std(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window_seconds: int,
) -> FloatVector:
    del blocks
    starts = helpers.time_window_bounds(series.timestamps, window_seconds=window_seconds)
    return helpers.time_rolling_std(resolved_dependencies[dependency_name], starts)


def _trend_slope_600s(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks
    return helpers.time_trend_slope(
        resolved_dependencies["log_base_fee"],
        series.timestamps,
        window_seconds=600,
    )


FEATURE_FAMILY_SPEC = FeatureFamilySpec(
    id="time_native",
    config_type=TimeNativeFeatureFamilyConfig,
    features={
        "log_base_fee": FeatureDefinition("log_base_fee", (), 0, 0, helpers.log_base_fee_feature),
        "gas_utilization": FeatureDefinition(
            "gas_utilization", (), 0, 0, helpers.gas_utilization_feature
        ),
        "seconds_since_previous_block": FeatureDefinition(
            "seconds_since_previous_block",
            (),
            0,
            0,
            _seconds_since_previous_block,
        ),
        "elapsed_seconds": FeatureDefinition("elapsed_seconds", (), 0, 0, _elapsed_seconds),
        "hour_sin": FeatureDefinition("hour_sin", (), 0, 0, helpers.hour_sin_feature),
        "hour_cos": FeatureDefinition("hour_cos", (), 0, 0, helpers.hour_cos_feature),
        "weekday_sin": FeatureDefinition("weekday_sin", (), 0, 0, helpers.weekday_sin_feature),
        "weekday_cos": FeatureDefinition("weekday_cos", (), 0, 0, helpers.weekday_cos_feature),
        "rolling_mean_log_base_fee_60s": FeatureDefinition(
            "rolling_mean_log_base_fee_60s",
            ("log_base_fee",),
            60,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=60,
            ),
        ),
        "rolling_std_log_base_fee_60s": FeatureDefinition(
            "rolling_std_log_base_fee_60s",
            ("log_base_fee",),
            60,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=60,
            ),
        ),
        "rolling_mean_gas_utilization_60s": FeatureDefinition(
            "rolling_mean_gas_utilization_60s",
            ("gas_utilization",),
            60,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=60,
            ),
        ),
        "rolling_std_gas_utilization_60s": FeatureDefinition(
            "rolling_std_gas_utilization_60s",
            ("gas_utilization",),
            60,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=60,
            ),
        ),
        "rolling_mean_log_base_fee_300s": FeatureDefinition(
            "rolling_mean_log_base_fee_300s",
            ("log_base_fee",),
            300,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=300,
            ),
        ),
        "rolling_std_log_base_fee_300s": FeatureDefinition(
            "rolling_std_log_base_fee_300s",
            ("log_base_fee",),
            300,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=300,
            ),
        ),
        "rolling_mean_gas_utilization_300s": FeatureDefinition(
            "rolling_mean_gas_utilization_300s",
            ("gas_utilization",),
            300,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=300,
            ),
        ),
        "rolling_std_gas_utilization_300s": FeatureDefinition(
            "rolling_std_gas_utilization_300s",
            ("gas_utilization",),
            300,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=300,
            ),
        ),
        "rolling_mean_log_base_fee_600s": FeatureDefinition(
            "rolling_mean_log_base_fee_600s",
            ("log_base_fee",),
            600,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=600,
            ),
        ),
        "rolling_std_log_base_fee_600s": FeatureDefinition(
            "rolling_std_log_base_fee_600s",
            ("log_base_fee",),
            600,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window_seconds=600,
            ),
        ),
        "rolling_mean_gas_utilization_600s": FeatureDefinition(
            "rolling_mean_gas_utilization_600s",
            ("gas_utilization",),
            600,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=600,
            ),
        ),
        "rolling_std_gas_utilization_600s": FeatureDefinition(
            "rolling_std_gas_utilization_600s",
            ("gas_utilization",),
            600,
            0,
            lambda blocks, series, resolved_dependencies: _time_rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window_seconds=600,
            ),
        ),
        "trend_slope_600s": FeatureDefinition(
            "trend_slope_600s",
            ("log_base_fee",),
            600,
            0,
            _trend_slope_600s,
        ),
    },
    fingerprint_sources=(Path(__file__).resolve(), Path(helpers.__file__).resolve()),
    build_series=helpers.build_canonical_series,
)

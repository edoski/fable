"""Time-native feature family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ..core import CanonicalBlockSeries
from . import helpers
from .base import FeatureDefinition, FeatureFamily, FeatureFamilyConfig

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


def _rolling_feature(
    *,
    dependency_name: str,
    stat: str,
    window_seconds: int,
) -> FeatureDefinition:
    compute = _time_rolling_mean if stat == "mean" else _time_rolling_std
    return FeatureDefinition(
        (dependency_name,),
        window_seconds,
        0,
        lambda blocks, series, resolved_dependencies, dependency_name=dependency_name, window_seconds=window_seconds, compute=compute: compute(
            blocks,
            series,
            resolved_dependencies,
            dependency_name=dependency_name,
            window_seconds=window_seconds,
        ),
    )


def _feature_definitions() -> dict[str, FeatureDefinition]:
    features: dict[str, FeatureDefinition] = {
        "log_base_fee": FeatureDefinition((), 0, 0, helpers.log_base_fee_feature),
        "gas_utilization": FeatureDefinition((), 0, 0, helpers.gas_utilization_feature),
        "seconds_since_previous_block": FeatureDefinition(
            (),
            0,
            0,
            _seconds_since_previous_block,
        ),
        "elapsed_seconds": FeatureDefinition((), 0, 0, _elapsed_seconds),
        "hour_sin": FeatureDefinition((), 0, 0, helpers.hour_sin_feature),
        "hour_cos": FeatureDefinition((), 0, 0, helpers.hour_cos_feature),
        "weekday_sin": FeatureDefinition((), 0, 0, helpers.weekday_sin_feature),
        "weekday_cos": FeatureDefinition((), 0, 0, helpers.weekday_cos_feature),
        "trend_slope_600s": FeatureDefinition(
            ("log_base_fee",),
            600,
            0,
            _trend_slope_600s,
        ),
    }
    for window_seconds in (60, 300, 600):
        for dependency_name in ("log_base_fee", "gas_utilization"):
            for stat in ("mean", "std"):
                name = f"rolling_{stat}_{dependency_name}_{window_seconds}s"
                features[name] = _rolling_feature(
                    dependency_name=dependency_name,
                    stat=stat,
                    window_seconds=window_seconds,
                )
    return features


TIME_NATIVE_FAMILY = FeatureFamily(
    features=_feature_definitions(),
    fingerprint_sources=(Path(__file__).resolve(), Path(helpers.__file__).resolve()),
    build_series=helpers.build_canonical_series,
)

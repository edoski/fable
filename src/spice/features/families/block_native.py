"""Block-native feature family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ..core import CanonicalBlockSeries
from . import helpers
from .base import FeatureDefinition, FeatureFamilyConfig, FeatureFamilySpec

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


def _rolling_mean(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window: int,
) -> FloatVector:
    del blocks, series
    return helpers.block_rolling_stat(
        resolved_dependencies[dependency_name],
        window=window,
        stat="mean",
    )


def _rolling_std(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
    *,
    dependency_name: str,
    window: int,
) -> FloatVector:
    del blocks, series
    return helpers.block_rolling_stat(
        resolved_dependencies[dependency_name],
        window=window,
        stat="std",
    )


def _trend_slope_200(
    blocks: pl.DataFrame,
    series: CanonicalBlockSeries,
    resolved_dependencies: Mapping[str, FloatVector],
) -> FloatVector:
    del blocks, series
    return helpers.block_trend_slope(resolved_dependencies["log_base_fee"], window=200)


FEATURE_FAMILY_SPEC = FeatureFamilySpec(
    id="block_native",
    config_type=BlockNativeFeatureFamilyConfig,
    features={
        "log_base_fee": FeatureDefinition("log_base_fee", (), 0, 0, helpers.log_base_fee_feature),
        "gas_utilization": FeatureDefinition(
            "gas_utilization", (), 0, 0, helpers.gas_utilization_feature
        ),
        "hour_sin": FeatureDefinition("hour_sin", (), 0, 0, helpers.hour_sin_feature),
        "hour_cos": FeatureDefinition("hour_cos", (), 0, 0, helpers.hour_cos_feature),
        "weekday_sin": FeatureDefinition("weekday_sin", (), 0, 0, helpers.weekday_sin_feature),
        "weekday_cos": FeatureDefinition("weekday_cos", (), 0, 0, helpers.weekday_cos_feature),
        "elapsed_blocks": FeatureDefinition("elapsed_blocks", (), 0, 0, _elapsed_blocks),
        "rolling_mean_log_base_fee_10": FeatureDefinition(
            "rolling_mean_log_base_fee_10",
            ("log_base_fee",),
            0,
            9,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=10,
            ),
        ),
        "rolling_std_log_base_fee_10": FeatureDefinition(
            "rolling_std_log_base_fee_10",
            ("log_base_fee",),
            0,
            9,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=10,
            ),
        ),
        "rolling_mean_gas_utilization_10": FeatureDefinition(
            "rolling_mean_gas_utilization_10",
            ("gas_utilization",),
            0,
            9,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=10,
            ),
        ),
        "rolling_std_gas_utilization_10": FeatureDefinition(
            "rolling_std_gas_utilization_10",
            ("gas_utilization",),
            0,
            9,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=10,
            ),
        ),
        "rolling_mean_log_base_fee_50": FeatureDefinition(
            "rolling_mean_log_base_fee_50",
            ("log_base_fee",),
            0,
            49,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=50,
            ),
        ),
        "rolling_std_log_base_fee_50": FeatureDefinition(
            "rolling_std_log_base_fee_50",
            ("log_base_fee",),
            0,
            49,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=50,
            ),
        ),
        "rolling_mean_gas_utilization_50": FeatureDefinition(
            "rolling_mean_gas_utilization_50",
            ("gas_utilization",),
            0,
            49,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=50,
            ),
        ),
        "rolling_std_gas_utilization_50": FeatureDefinition(
            "rolling_std_gas_utilization_50",
            ("gas_utilization",),
            0,
            49,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=50,
            ),
        ),
        "rolling_mean_log_base_fee_200": FeatureDefinition(
            "rolling_mean_log_base_fee_200",
            ("log_base_fee",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=200,
            ),
        ),
        "rolling_std_log_base_fee_200": FeatureDefinition(
            "rolling_std_log_base_fee_200",
            ("log_base_fee",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="log_base_fee",
                window=200,
            ),
        ),
        "rolling_mean_gas_utilization_200": FeatureDefinition(
            "rolling_mean_gas_utilization_200",
            ("gas_utilization",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _rolling_mean(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=200,
            ),
        ),
        "rolling_std_gas_utilization_200": FeatureDefinition(
            "rolling_std_gas_utilization_200",
            ("gas_utilization",),
            0,
            199,
            lambda blocks, series, resolved_dependencies: _rolling_std(
                blocks,
                series,
                resolved_dependencies,
                dependency_name="gas_utilization",
                window=200,
            ),
        ),
        "trend_slope_200": FeatureDefinition(
            "trend_slope_200",
            ("log_base_fee",),
            0,
            199,
            _trend_slope_200,
        ),
    },
    fingerprint_sources=(Path(__file__).resolve(), Path(helpers.__file__).resolve()),
    build_series=helpers.build_canonical_series,
)

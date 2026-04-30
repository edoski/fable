"""Safe fee-dynamics feature catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ..core import (
    CanonicalBlockSeries,
    FeatureCatalog,
    FeatureSpec,
    FloatVector,
    SourceSpec,
    dow_cos,
    dow_sin,
    hour_cos,
    hour_sin,
    rolling_stat,
    shift,
    shifted_column,
)


def _float_column(blocks: pl.DataFrame, column: str) -> FloatVector:
    return blocks[column].cast(pl.Float64).to_numpy().astype(np.float64, copy=False)


def _log1p(values: FloatVector) -> FloatVector:
    return np.log1p(np.clip(values, 0.0, None))


def _log_source(sources: Mapping[str, FloatVector], name: str) -> FloatVector:
    return np.log(np.clip(sources[name], 1.0, None))


def _delta(values: FloatVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size > 1:
        result[1:] = np.diff(values)
    return result


def _binary_trend(values: FloatVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    result[valid] = np.where(values[valid] >= 0.0, 1.0, -1.0)
    return result


def _gas_utilization(values_used: FloatVector, values_limit: FloatVector) -> FloatVector:
    result = np.full(values_used.shape[0], np.nan, dtype=np.float64)
    valid = values_limit > 0.0
    result[valid] = values_used[valid] / values_limit[valid]
    return result


def _elapsed_seconds(series: CanonicalBlockSeries) -> FloatVector:
    if series.timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    return series.timestamps.astype(np.float64, copy=False) - float(series.timestamps[0])


def _dt_seconds(series: CanonicalBlockSeries) -> FloatVector:
    if series.timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.empty(series.timestamps.shape[0], dtype=np.float64)
    result[0] = 0.0
    if series.timestamps.size > 1:
        result[1:] = np.diff(series.timestamps.astype(np.float64, copy=False))
    return result


def _sources() -> dict[str, SourceSpec]:
    return {
        # EIP-1559 base fee for block t is deterministic from parent state and known
        # before block t execution, so it is safe as a current-row source.
        "current_base_fee_per_gas": SourceSpec(
            source_columns=("base_fee_per_gas",),
            warmup_rows=0,
            required_after_warmup=True,
            compute=lambda blocks: _float_column(blocks, "base_fee_per_gas"),
        ),
        "prev_gas_used": SourceSpec(
            source_columns=("gas_used",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "gas_used"),
        ),
        "prev_gas_limit": SourceSpec(
            source_columns=("gas_limit",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "gas_limit"),
        ),
        "prev_tx_count": SourceSpec(
            source_columns=("tx_count",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "tx_count"),
        ),
        "prev_priority_fee_p10": SourceSpec(
            source_columns=("priority_fee_p10",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p10"),
        ),
        "prev_priority_fee_p50": SourceSpec(
            source_columns=("priority_fee_p50",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p50"),
        ),
        "prev_priority_fee_p90": SourceSpec(
            source_columns=("priority_fee_p90",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p90"),
        ),
        "prev_priority_fee_spread": SourceSpec(
            source_columns=("priority_fee_spread",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_spread"),
        ),
        "prev_fee_history_gas_used_ratio": SourceSpec(
            source_columns=("fee_history_gas_used_ratio",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "fee_history_gas_used_ratio"),
        ),
    }


def _features() -> dict[str, FeatureSpec]:
    features: dict[str, FeatureSpec] = {
        "log_base_fee_per_gas": FeatureSpec(
            source_dependencies=("current_base_fee_per_gas",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: _log_source(
                sources,
                "current_base_fee_per_gas",
            ),
        ),
        "log_prev_gas_used": FeatureSpec(
            source_dependencies=("prev_gas_used",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _log1p(sources["prev_gas_used"]),
        ),
        "log_prev_gas_limit": FeatureSpec(
            source_dependencies=("prev_gas_limit",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _log1p(sources["prev_gas_limit"]),
        ),
        "prev_gas_utilization": FeatureSpec(
            source_dependencies=("prev_gas_used", "prev_gas_limit"),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _gas_utilization(
                sources["prev_gas_used"],
                sources["prev_gas_limit"],
            ),
        ),
        "prev_eip1559_pressure": FeatureSpec(
            source_dependencies=("prev_gas_used", "prev_gas_limit"),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _gas_utilization(
                sources["prev_gas_used"],
                sources["prev_gas_limit"] / 2.0,
            ),
        ),
        "log_prev_tx_count": FeatureSpec(
            source_dependencies=("prev_tx_count",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _log1p(sources["prev_tx_count"]),
        ),
        "seconds_since_previous_block": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _dt_seconds(series),
        ),
        "elapsed_seconds": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: _elapsed_seconds(series),
        ),
        "hour_sin": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: hour_sin(series.timestamps),
        ),
        "hour_cos": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: hour_cos(series.timestamps),
        ),
        "dow_sin": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: dow_sin(series.timestamps),
        ),
        "dow_cos": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=0,
            compute=lambda blocks, series, sources, features: dow_cos(series.timestamps),
        ),
        "prev_priority_fee_p10": FeatureSpec(
            source_dependencies=("prev_priority_fee_p10",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: sources["prev_priority_fee_p10"],
        ),
        "prev_priority_fee_p50": FeatureSpec(
            source_dependencies=("prev_priority_fee_p50",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: sources["prev_priority_fee_p50"],
        ),
        "prev_priority_fee_p90": FeatureSpec(
            source_dependencies=("prev_priority_fee_p90",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: sources["prev_priority_fee_p90"],
        ),
        "prev_priority_fee_spread": FeatureSpec(
            source_dependencies=("prev_priority_fee_spread",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: sources["prev_priority_fee_spread"],
        ),
        "prev_fee_history_gas_used_ratio": FeatureSpec(
            source_dependencies=("prev_fee_history_gas_used_ratio",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: sources[
                "prev_fee_history_gas_used_ratio"
            ],
        ),
        "dlog_base_fee": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _delta(
                features["log_base_fee_per_gas"]
            ),
        ),
        "base_fee_trend": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("dlog_base_fee",),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: _binary_trend(
                features["dlog_base_fee"]
            ),
        ),
    }
    for window in (25, 100):
        features[f"roll{window}_mean_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="mean",
            ),
        )
        features[f"roll{window}_std_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="std",
            ),
        )
        features[f"roll{window}_min_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="min",
            ),
        )
    for window in (10, 50, 200):
        features[f"roll{window}_mean_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="mean",
            ),
        )
        features[f"roll{window}_std_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="std",
                ddof=1,
            ),
        )
        features[f"roll{window}_min_logfee"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_base_fee_per_gas",),
            history_seconds=0,
            warmup_rows=window - 1,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["log_base_fee_per_gas"],
                window=window,
                stat="min",
            ),
        )
        features[f"roll{window}_mean_prev_gas_utilization"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("prev_gas_utilization",),
            history_seconds=0,
            warmup_rows=window,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["prev_gas_utilization"],
                window=window,
                stat="mean",
            ),
        )
        features[f"roll{window}_std_prev_gas_utilization"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("prev_gas_utilization",),
            history_seconds=0,
            warmup_rows=window,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features["prev_gas_utilization"],
                window=window,
                stat="std",
                ddof=1,
            ),
        )
    for lag in range(1, 7):
        features[f"dlog_base_fee_lag{lag}"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("dlog_base_fee",),
            history_seconds=0,
            warmup_rows=lag + 1,
            compute=lambda blocks, series, sources, features, lag=lag: shift(
                features["dlog_base_fee"],
                lag=lag,
            ),
        )
        features[f"prev_gas_utilization_lag{lag}"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("prev_gas_utilization",),
            history_seconds=0,
            warmup_rows=lag + 1,
            compute=lambda blocks, series, sources, features, lag=lag: shift(
                features["prev_gas_utilization"],
                lag=lag,
            ),
        )
    return features


CORE_FEE_DYNAMICS = FeatureCatalog(
    sources=_sources(),
    features=_features(),
    fingerprint_sources=(
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "core.py",
    ),
)

CORE_FEE_DYNAMICS_OUTPUTS = (
    "log_base_fee_per_gas",
    "log_prev_gas_used",
    "log_prev_gas_limit",
    "prev_gas_utilization",
    "prev_eip1559_pressure",
    "log_prev_tx_count",
    "seconds_since_previous_block",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "roll25_mean_logfee",
    "roll25_std_logfee",
    "roll25_min_logfee",
    "roll100_mean_logfee",
    "roll100_std_logfee",
    "roll100_min_logfee",
    "prev_priority_fee_p10",
    "prev_priority_fee_p50",
    "prev_priority_fee_p90",
    "prev_priority_fee_spread",
    "prev_fee_history_gas_used_ratio",
    "dlog_base_fee",
    "base_fee_trend",
    *(f"dlog_base_fee_lag{lag}" for lag in range(1, 7)),
    *(f"prev_gas_utilization_lag{lag}" for lag in range(1, 7)),
    "roll10_mean_logfee",
    "roll10_std_logfee",
    "roll10_min_logfee",
    "roll50_mean_logfee",
    "roll50_std_logfee",
    "roll50_min_logfee",
    "roll200_mean_logfee",
    "roll200_std_logfee",
    "roll200_min_logfee",
    "roll10_mean_prev_gas_utilization",
    "roll10_std_prev_gas_utilization",
    "roll50_mean_prev_gas_utilization",
    "roll50_std_prev_gas_utilization",
    "roll200_mean_prev_gas_utilization",
    "roll200_std_prev_gas_utilization",
)
CORE_FEE_DYNAMICS_ELAPSED_POSITION_OUTPUTS = CORE_FEE_DYNAMICS_OUTPUTS + (
    "elapsed_seconds",
)

"""Block fact sources and gas-utilization features."""

from __future__ import annotations

from ...core import FeatureSpec, SourceSpec
from ._transforms import gas_utilization, log1p, rolling_stat, shift, shifted_column

PREVIOUS_BLOCK_FACT_OUTPUTS = (
    "log_prev_gas_used",
    "log_prev_gas_limit",
    "prev_gas_utilization",
    "log_prev_tx_count",
)
PREVIOUS_GAS_UTILIZATION_TREND_OUTPUTS = (
    *(f"prev_gas_utilization_lag{lag}" for lag in range(1, 7)),
)
PREVIOUS_GAS_UTILIZATION_ROLLING_OUTPUTS = (
    "roll10_mean_prev_gas_utilization",
    "roll10_std_prev_gas_utilization",
    "roll50_mean_prev_gas_utilization",
    "roll50_std_prev_gas_utilization",
    "roll200_mean_prev_gas_utilization",
    "roll200_std_prev_gas_utilization",
)


def previous_block_fact_sources() -> dict[str, SourceSpec]:
    return {
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
    }


def previous_block_fact_features() -> dict[str, FeatureSpec]:
    return {
        "log_prev_gas_used": FeatureSpec(
            source_dependencies=("prev_gas_used",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: log1p(sources["prev_gas_used"]),
        ),
        "log_prev_gas_limit": FeatureSpec(
            source_dependencies=("prev_gas_limit",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: log1p(sources["prev_gas_limit"]),
        ),
        "prev_gas_utilization": FeatureSpec(
            source_dependencies=("prev_gas_used", "prev_gas_limit"),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: gas_utilization(
                sources["prev_gas_used"],
                sources["prev_gas_limit"],
            ),
        ),
        "log_prev_tx_count": FeatureSpec(
            source_dependencies=("prev_tx_count",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: log1p(sources["prev_tx_count"]),
        ),
    }


def gas_utilization_trend_features(
    feature_name: str,
    *,
    base_warmup_rows: int,
) -> dict[str, FeatureSpec]:
    return {
        f"{feature_name}_lag{lag}": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(feature_name,),
            history_seconds=0,
            warmup_rows=lag + base_warmup_rows,
            compute=lambda blocks, series, sources, features, lag=lag: shift(
                features[feature_name],
                lag=lag,
            ),
        )
        for lag in range(1, 7)
    }


def gas_utilization_rolling_features(
    feature_name: str,
    *,
    base_warmup_rows: int,
) -> dict[str, FeatureSpec]:
    features: dict[str, FeatureSpec] = {}
    for window in (10, 50, 200):
        features[f"roll{window}_mean_{feature_name}"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(feature_name,),
            history_seconds=0,
            warmup_rows=window - 1 + base_warmup_rows,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features[feature_name],
                window=window,
                stat="mean",
            ),
        )
        features[f"roll{window}_std_{feature_name}"] = FeatureSpec(
            source_dependencies=(),
            feature_dependencies=(feature_name,),
            history_seconds=0,
            warmup_rows=window - 1 + base_warmup_rows,
            compute=lambda blocks, series, sources, features, window=window: rolling_stat(
                features[feature_name],
                window=window,
                stat="std",
                ddof=1,
            ),
        )
    return features

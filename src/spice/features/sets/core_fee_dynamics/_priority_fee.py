"""Priority-fee sources and features."""

from __future__ import annotations

from ...core import FeatureSpec, SourceSpec
from ._transforms import delta, log1p, rolling_feature, shift_feature, shifted_column

PRIORITY_FEE_PERCENTILES_ENRICHMENT = "priority_fee_percentiles"
PRIORITY_FEE_OUTPUTS = (
    "prev_priority_fee_p10",
    "prev_priority_fee_p50",
    "prev_priority_fee_p90",
    "prev_priority_fee_spread",
    "log_prev_priority_fee_p50",
    "dlog_prev_priority_fee_p50",
    *(f"dlog_prev_priority_fee_p50_lag{lag}" for lag in range(1, 7)),
    "roll10_mean_log_prev_priority_fee_p50",
    "roll10_std_log_prev_priority_fee_p50",
    "roll50_mean_log_prev_priority_fee_p50",
    "roll50_std_log_prev_priority_fee_p50",
    "roll200_mean_log_prev_priority_fee_p50",
    "roll200_std_log_prev_priority_fee_p50",
    "log_prev_priority_fee_spread",
    "dlog_prev_priority_fee_spread",
    *(f"dlog_prev_priority_fee_spread_lag{lag}" for lag in range(1, 7)),
    "roll10_mean_log_prev_priority_fee_spread",
    "roll10_std_log_prev_priority_fee_spread",
    "roll50_mean_log_prev_priority_fee_spread",
    "roll50_std_log_prev_priority_fee_spread",
    "roll200_mean_log_prev_priority_fee_spread",
    "roll200_std_log_prev_priority_fee_spread",
)


def priority_fee_sources() -> dict[str, SourceSpec]:
    return {
        "prev_priority_fee_p10": SourceSpec(
            source_columns=("priority_fee_p10",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p10"),
            optional_enrichments=frozenset({PRIORITY_FEE_PERCENTILES_ENRICHMENT}),
        ),
        "prev_priority_fee_p50": SourceSpec(
            source_columns=("priority_fee_p50",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p50"),
            optional_enrichments=frozenset({PRIORITY_FEE_PERCENTILES_ENRICHMENT}),
        ),
        "prev_priority_fee_p90": SourceSpec(
            source_columns=("priority_fee_p90",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_p90"),
            optional_enrichments=frozenset({PRIORITY_FEE_PERCENTILES_ENRICHMENT}),
        ),
        "prev_priority_fee_spread": SourceSpec(
            source_columns=("priority_fee_spread",),
            warmup_rows=1,
            required_after_warmup=True,
            compute=lambda blocks: shifted_column(blocks, "priority_fee_spread"),
            optional_enrichments=frozenset({PRIORITY_FEE_PERCENTILES_ENRICHMENT}),
        ),
    }


def priority_fee_features() -> dict[str, FeatureSpec]:
    features: dict[str, FeatureSpec] = {
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
            compute=lambda blocks, series, sources, features: sources[
                "prev_priority_fee_spread"
            ],
        ),
        "log_prev_priority_fee_p50": FeatureSpec(
            source_dependencies=("prev_priority_fee_p50",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: log1p(
                sources["prev_priority_fee_p50"]
            ),
        ),
        "log_prev_priority_fee_spread": FeatureSpec(
            source_dependencies=("prev_priority_fee_spread",),
            feature_dependencies=(),
            history_seconds=0,
            warmup_rows=1,
            compute=lambda blocks, series, sources, features: log1p(
                sources["prev_priority_fee_spread"]
            ),
        ),
        "dlog_prev_priority_fee_p50": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_prev_priority_fee_p50",),
            history_seconds=0,
            warmup_rows=2,
            compute=lambda blocks, series, sources, features: delta(
                features["log_prev_priority_fee_p50"]
            ),
        ),
        "dlog_prev_priority_fee_spread": FeatureSpec(
            source_dependencies=(),
            feature_dependencies=("log_prev_priority_fee_spread",),
            history_seconds=0,
            warmup_rows=2,
            compute=lambda blocks, series, sources, features: delta(
                features["log_prev_priority_fee_spread"]
            ),
        ),
    }
    for window in (10, 50, 200):
        for priority_name in ("p50", "spread"):
            log_feature = f"log_prev_priority_fee_{priority_name}"
            features[f"roll{window}_mean_{log_feature}"] = FeatureSpec(
                source_dependencies=(),
                feature_dependencies=(log_feature,),
                history_seconds=0,
                warmup_rows=window,
                compute=rolling_feature(log_feature, window=window, stat="mean"),
            )
            features[f"roll{window}_std_{log_feature}"] = FeatureSpec(
                source_dependencies=(),
                feature_dependencies=(log_feature,),
                history_seconds=0,
                warmup_rows=window,
                compute=rolling_feature(log_feature, window=window, stat="std", ddof=1),
            )
    for lag in range(1, 7):
        for priority_name in ("p50", "spread"):
            dlog_feature = f"dlog_prev_priority_fee_{priority_name}"
            features[f"{dlog_feature}_lag{lag}"] = FeatureSpec(
                source_dependencies=(),
                feature_dependencies=(dlog_feature,),
                history_seconds=0,
                warmup_rows=lag + 2,
                compute=shift_feature(dlog_feature, lag=lag),
            )
    return features

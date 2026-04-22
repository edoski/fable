from __future__ import annotations

import polars as pl

from spice.reconstruction.features import materialize_feature_candidate
from spice.reconstruction.models import FeatureCandidate


def _blocks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "block_number": [10, 11, 12, 13],
            "timestamp": [100, 112, 124, 136],
            "gas_used": [50, 100, 75, 90],
            "gas_limit": [100, 100, 100, 100],
            "base_fee_per_gas": [10.0, 12.0, 9.0, 11.0],
            "block_usage_ratio": [0.5, 1.0, 0.75, 0.9],
        }
    )


def test_materialize_feature_candidate_supports_reference_like_formula_choices() -> None:
    materialized, summary = materialize_feature_candidate(
        _blocks(),
        chain="ethereum",
        delay_seconds=36,
        candidate=FeatureCandidate(
            gas_ratio_mode="derived_percent",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_prev_delta_sign",
        ),
    )

    assert materialized["gas_ratio"].to_list() == [50.0, 100.0, 75.0, 90.0]
    assert materialized["time_since_start"].to_list() == [0.0, 12.0, 24.0, 36.0]
    assert tuple(summary.base_fee_trend_unique) == (-1.0, 1.0)
    assert summary.score < 1.0

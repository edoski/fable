from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest

from spice.config import coerce_features_config
from spice.config.registry import load_named_group
from spice.features import (
    CompiledFeatureContract,
    FeaturePrerequisites,
    compile_feature_contract,
    validate_feature_selection,
)
from spice.features import core as feature_core
from spice.features.sets import core_fee_dynamics as core_fee_dynamics_module
from spice.features.sets.core_fee_dynamics import CORE_FEE_DYNAMICS

BASELINE_OUTPUTS = [
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
    "prev_fee_history_gas_used_ratio",
    "dlog_base_fee",
    "base_fee_trend",
    *[f"dlog_base_fee_lag{lag}" for lag in range(1, 7)],
    *[f"prev_gas_utilization_lag{lag}" for lag in range(1, 7)],
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
]
UNSAFE_OUTPUTS = [
    "log_base_fee_per_gas",
    "log_current_gas_used",
    "log_current_gas_limit",
    "current_gas_utilization",
    "current_eip1559_pressure",
    "log_current_tx_count",
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
    "current_fee_history_gas_used_ratio",
    "dlog_base_fee",
    "base_fee_trend",
    *[f"dlog_base_fee_lag{lag}" for lag in range(1, 7)],
    *[f"current_gas_utilization_lag{lag}" for lag in range(1, 7)],
    "roll10_mean_logfee",
    "roll10_std_logfee",
    "roll10_min_logfee",
    "roll50_mean_logfee",
    "roll50_std_logfee",
    "roll50_min_logfee",
    "roll200_mean_logfee",
    "roll200_std_logfee",
    "roll200_min_logfee",
    "roll10_mean_current_gas_utilization",
    "roll10_std_current_gas_utilization",
    "roll50_mean_current_gas_utilization",
    "roll50_std_current_gas_utilization",
    "roll200_mean_current_gas_utilization",
    "roll200_std_current_gas_utilization",
]
PRIORITY_FEE_OUTPUTS = [
    "prev_priority_fee_p10",
    "prev_priority_fee_p50",
    "prev_priority_fee_p90",
    "prev_priority_fee_spread",
    "log_prev_priority_fee_p50",
    "dlog_prev_priority_fee_p50",
    *[f"dlog_prev_priority_fee_p50_lag{lag}" for lag in range(1, 7)],
    "roll10_mean_log_prev_priority_fee_p50",
    "roll10_std_log_prev_priority_fee_p50",
    "roll50_mean_log_prev_priority_fee_p50",
    "roll50_std_log_prev_priority_fee_p50",
    "roll200_mean_log_prev_priority_fee_p50",
    "roll200_std_log_prev_priority_fee_p50",
    "log_prev_priority_fee_spread",
    "dlog_prev_priority_fee_spread",
    *[f"dlog_prev_priority_fee_spread_lag{lag}" for lag in range(1, 7)],
    "roll10_mean_log_prev_priority_fee_spread",
    "roll10_std_log_prev_priority_fee_spread",
    "roll50_mean_log_prev_priority_fee_spread",
    "roll50_std_log_prev_priority_fee_spread",
    "roll200_mean_log_prev_priority_fee_spread",
    "roll200_std_log_prev_priority_fee_spread",
]


def _frame(row_count: int = 140) -> pl.DataFrame:
    block_numbers = np.arange(10_000, 10_000 + row_count, dtype=np.int64)
    return pl.DataFrame(
        {
            "block_number": block_numbers,
            "timestamp": np.arange(row_count, dtype=np.int64) * 12,
            "base_fee_per_gas": 1_000_000_000 + block_numbers,
            "gas_used": 15_000_000 + np.arange(row_count, dtype=np.int64),
            "gas_limit": np.full(row_count, 30_000_000, dtype=np.int64),
            "tx_count": 100 + np.arange(row_count, dtype=np.int64),
            "block_size_bytes": [None] * row_count,
            "blob_gas_used": [None] * row_count,
            "excess_blob_gas": [None] * row_count,
            "priority_fee_p10": np.full(row_count, 1_000_000, dtype=np.int64),
            "priority_fee_p50": np.full(row_count, 2_000_000, dtype=np.int64),
            "priority_fee_p90": np.full(row_count, 4_000_000, dtype=np.int64),
            "priority_fee_spread": np.full(row_count, 3_000_000, dtype=np.int64),
            "fee_history_gas_used_ratio": np.full(row_count, 0.5, dtype=np.float64),
            "chain_id": np.ones(row_count, dtype=np.int64),
        }
    ).sample(fraction=1.0, shuffle=True, seed=7)


def _contract(outputs: list[str], *, features_id: str = "core_fee_dynamics"):
    return compile_feature_contract(
        features=coerce_features_config({"id": features_id, "outputs": outputs})
    )


def test_core_fee_dynamics_builds_finite_aligned_feature_table() -> None:
    contract = _contract(
        [
            "log_base_fee_per_gas",
            "log_prev_gas_used",
            "prev_gas_utilization",
            "roll100_mean_logfee",
            "prev_fee_history_gas_used_ratio",
        ]
    )

    table = contract.build_table(_frame())

    assert contract.feature_prerequisites == FeaturePrerequisites(warmup_rows=99)
    assert table.feature_matrix.shape == (140, 5)
    assert np.isfinite(table.feature_matrix).all()
    assert table.series.block_numbers[0] == 10_000
    assert table.feature_matrix[0, 1] == 0.0
    assert table.feature_matrix[99, 4] == pytest.approx(0.5)


def test_core_fee_dynamics_lags_finalized_current_block_facts() -> None:
    table = _contract(
        [
            "log_prev_gas_used",
            "prev_fee_history_gas_used_ratio",
        ]
    ).build_table(_frame(4))

    assert table.feature_matrix[1, 0] == pytest.approx(np.log1p(15_000_000))
    assert table.feature_matrix[1, 1] == pytest.approx(0.5)


def test_core_fee_dynamics_requires_base_fee_after_warmup() -> None:
    frame = _frame(4).with_columns(
        pl.when(pl.col("block_number") == 10_003)
        .then(None)
        .otherwise(pl.col("base_fee_per_gas"))
        .alias("base_fee_per_gas")
    )

    with pytest.raises(ValueError, match="current_base_fee_per_gas"):
        _contract(["log_base_fee_per_gas"]).build_table(frame)


def test_core_fee_dynamics_validates_sources_before_global_feature_warmup() -> None:
    frame = _frame(140).with_columns(
        pl.when(pl.col("block_number") == 10_000)
        .then(None)
        .otherwise(pl.col("base_fee_per_gas"))
        .alias("base_fee_per_gas")
    )

    with pytest.raises(ValueError, match="current_base_fee_per_gas"):
        _contract(["log_base_fee_per_gas", "roll100_mean_logfee"]).build_table(frame)


def test_core_fee_dynamics_requires_lagged_fee_history_after_warmup() -> None:
    frame = _frame(4).with_columns(
        pl.when(pl.col("block_number") == 10_001)
        .then(None)
        .otherwise(pl.col("fee_history_gas_used_ratio"))
        .alias("fee_history_gas_used_ratio")
    )

    with pytest.raises(ValueError, match="prev_fee_history_gas_used_ratio"):
        _contract(["prev_fee_history_gas_used_ratio"]).build_table(frame)


def test_feature_selection_validation_rejects_unknown_outputs() -> None:
    with pytest.raises(ValueError, match="Unknown feature outputs"):
        validate_feature_selection("core_fee_dynamics", ("raw_producer_address",))


def test_core_fee_dynamics_rejects_priority_fee_outputs() -> None:
    with pytest.raises(ValueError, match="Unknown feature outputs: prev_priority_fee_p50"):
        coerce_features_config(
            {
                "id": "core_fee_dynamics",
                "outputs": [*BASELINE_OUTPUTS, "prev_priority_fee_p50"],
            }
        )

    coerce_features_config(
        {
            "id": "core_fee_dynamics_with_priority_fee",
            "outputs": [*BASELINE_OUTPUTS, "prev_priority_fee_p50"],
        }
    )


def test_core_fee_dynamics_fingerprint_includes_shared_engine() -> None:
    assert core_fee_dynamics_module.__file__ is not None
    assert feature_core.__file__ is not None
    assert CORE_FEE_DYNAMICS.fingerprint_sources == (
        Path(core_fee_dynamics_module.__file__).resolve(),
        Path(feature_core.__file__).resolve(),
    )


def test_default_core_fee_dynamics_excludes_time_since_start_signal() -> None:
    payload = cast(dict[str, object], load_named_group("core_fee_dynamics", "features"))
    outputs = cast(list[str], payload["outputs"])

    assert "elapsed_seconds" not in outputs
    with pytest.raises(ValueError, match="Unknown feature outputs: elapsed_seconds"):
        coerce_features_config(
            {
                "id": "core_fee_dynamics",
                "outputs": [*outputs, "elapsed_seconds"],
            }
        )


def test_compiled_contract_rejects_baseline_elapsed_position_output() -> None:
    contract = CompiledFeatureContract(
        features_id="core_fee_dynamics",
        feature_names=("elapsed_seconds",),
        feature_graph_fingerprint="manual",
        feature_prerequisites=FeaturePrerequisites(),
    )

    with pytest.raises(ValueError, match="Unknown feature outputs: elapsed_seconds"):
        contract.build_table(_frame())


def test_elapsed_position_ablation_config_adds_time_since_start_signal() -> None:
    baseline = cast(dict[str, object], load_named_group("core_fee_dynamics", "features"))
    ablation = cast(
        dict[str, object],
        load_named_group("core_fee_dynamics_elapsed_position", "features"),
    )
    baseline_outputs = cast(list[str], baseline["outputs"])
    ablation_outputs = cast(list[str], ablation["outputs"])

    assert ablation["id"] == "core_fee_dynamics_elapsed_position"
    assert ablation_outputs[:-1] == baseline_outputs
    assert ablation_outputs[-1] == "elapsed_seconds"
    coerce_features_config(ablation)


def test_core_fee_dynamics_config_includes_restored_safe_local_trends() -> None:
    baseline = cast(dict[str, object], load_named_group("core_fee_dynamics", "features"))
    baseline_outputs = cast(list[str], baseline["outputs"])

    assert baseline_outputs == BASELINE_OUTPUTS
    assert "elapsed_seconds" not in baseline_outputs
    assert "time_since_start" not in baseline_outputs
    assert "prev_priority_fee_p50" not in baseline_outputs
    assert "dlog_base_fee" in baseline_outputs
    assert "prev_gas_utilization_lag6" in baseline_outputs
    assert "roll200_std_prev_gas_utilization" in baseline_outputs
    coerce_features_config(baseline)


def test_core_fee_dynamics_restored_local_trends_build_finite_aligned_table() -> None:
    contract = _contract(
        [
            "dlog_base_fee",
            "base_fee_trend",
            "dlog_base_fee_lag6",
            "prev_gas_utilization_lag6",
            "roll200_mean_logfee",
            "roll200_std_prev_gas_utilization",
        ]
    )

    table = contract.build_table(_frame(220))

    assert contract.feature_prerequisites == FeaturePrerequisites(warmup_rows=200)
    assert table.feature_matrix.shape == (220, 6)
    assert np.isfinite(table.feature_matrix).all()
    assert table.feature_matrix[200, 1] == pytest.approx(1.0)


def test_core_fee_dynamics_local_trend_lags_use_prior_rows() -> None:
    table = _contract(
        [
            "prev_gas_utilization",
            "prev_gas_utilization_lag1",
            "dlog_base_fee",
            "dlog_base_fee_lag1",
        ]
    ).build_table(_frame(12))

    assert table.feature_matrix[3, 0] == pytest.approx((15_000_000 + 2) / 30_000_000)
    assert table.feature_matrix[3, 1] == pytest.approx((15_000_000 + 1) / 30_000_000)
    assert table.feature_matrix[3, 3] == pytest.approx(table.feature_matrix[2, 2])


def test_core_fee_dynamics_unsafe_config_replaces_same_block_facts() -> None:
    payload = cast(dict[str, object], load_named_group("core_fee_dynamics_unsafe", "features"))
    outputs = cast(list[str], payload["outputs"])

    assert payload["id"] == "core_fee_dynamics_unsafe"
    assert outputs == UNSAFE_OUTPUTS
    assert "log_prev_gas_used" not in outputs
    assert "log_current_gas_used" in outputs
    assert "prev_priority_fee_p50" not in outputs
    coerce_features_config(payload)


def test_core_fee_dynamics_unsafe_uses_same_block_facts() -> None:
    table = _contract(
        [
            "log_current_gas_used",
            "current_gas_utilization",
            "current_fee_history_gas_used_ratio",
            "current_gas_utilization_lag1",
        ],
        features_id="core_fee_dynamics_unsafe",
    ).build_table(_frame(12))

    assert table.feature_prerequisites == FeaturePrerequisites(warmup_rows=1)
    assert table.feature_matrix[3, 0] == pytest.approx(np.log1p(15_000_003))
    assert table.feature_matrix[3, 1] == pytest.approx((15_000_000 + 3) / 30_000_000)
    assert table.feature_matrix[3, 2] == pytest.approx(0.5)
    assert table.feature_matrix[3, 3] == pytest.approx((15_000_000 + 2) / 30_000_000)


def test_priority_fee_config_adds_scalar_and_trend_outputs() -> None:
    baseline = cast(dict[str, object], load_named_group("core_fee_dynamics", "features"))
    priority = cast(
        dict[str, object],
        load_named_group("core_fee_dynamics_with_priority_fee", "features"),
    )
    baseline_outputs = cast(list[str], baseline["outputs"])
    priority_outputs = cast(list[str], priority["outputs"])

    assert priority["id"] == "core_fee_dynamics_with_priority_fee"
    assert priority_outputs == [*baseline_outputs, *PRIORITY_FEE_OUTPUTS]
    assert "prev_priority_fee_p10" in priority_outputs
    assert "log_prev_priority_fee_p10" not in priority_outputs
    assert "elapsed_seconds" not in priority_outputs
    coerce_features_config(priority)


def test_priority_fee_features_build_finite_aligned_table() -> None:
    contract = _contract(
        [
            "prev_priority_fee_p50",
            "log_prev_priority_fee_p50",
            "dlog_prev_priority_fee_p50_lag6",
            "roll200_std_log_prev_priority_fee_p50",
            "log_prev_priority_fee_spread",
            "dlog_prev_priority_fee_spread_lag6",
            "roll200_mean_log_prev_priority_fee_spread",
        ],
        features_id="core_fee_dynamics_with_priority_fee",
    )

    table = contract.build_table(_frame(220))

    assert contract.feature_prerequisites == FeaturePrerequisites(warmup_rows=200)
    assert table.feature_matrix.shape == (220, 7)
    assert np.isfinite(table.feature_matrix).all()


def test_priority_fee_trend_lags_use_prior_fee_history_rows() -> None:
    frame = _frame(12).with_columns(
        (1_000 + (pl.col("block_number") - 10_000) * 10).alias("priority_fee_p50"),
        (500 + (pl.col("block_number") - 10_000) * 5).alias("priority_fee_spread"),
    )
    table = _contract(
        [
            "log_prev_priority_fee_p50",
            "dlog_prev_priority_fee_p50",
            "dlog_prev_priority_fee_p50_lag1",
            "log_prev_priority_fee_spread",
            "dlog_prev_priority_fee_spread",
            "dlog_prev_priority_fee_spread_lag1",
        ],
        features_id="core_fee_dynamics_with_priority_fee",
    ).build_table(frame)

    expected_p50_delta = np.log1p(1_030) - np.log1p(1_020)
    expected_spread_delta = np.log1p(515) - np.log1p(510)
    assert table.feature_matrix[4, 1] == pytest.approx(expected_p50_delta)
    assert table.feature_matrix[5, 2] == pytest.approx(expected_p50_delta)
    assert table.feature_matrix[4, 4] == pytest.approx(expected_spread_delta)
    assert table.feature_matrix[5, 5] == pytest.approx(expected_spread_delta)

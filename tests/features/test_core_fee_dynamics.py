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
    "prev_priority_fee_p10",
    "prev_priority_fee_p50",
    "prev_priority_fee_p90",
    "prev_priority_fee_spread",
    "prev_fee_history_gas_used_ratio",
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
            "prev_priority_fee_p50",
            "prev_fee_history_gas_used_ratio",
        ]
    )

    table = contract.build_table(_frame())

    assert contract.feature_prerequisites == FeaturePrerequisites(warmup_rows=99)
    assert table.feature_matrix.shape == (140, 6)
    assert np.isfinite(table.feature_matrix).all()
    assert table.series.block_numbers[0] == 10_000
    assert table.feature_matrix[0, 1] == 0.0
    assert table.feature_matrix[99, 4] == pytest.approx(2_000_000.0)


def test_core_fee_dynamics_lags_finalized_current_block_facts() -> None:
    table = _contract(
        [
            "log_prev_gas_used",
            "prev_priority_fee_p50",
            "prev_fee_history_gas_used_ratio",
        ]
    ).build_table(_frame(4))

    assert table.feature_matrix[1, 0] == pytest.approx(np.log1p(15_000_000))
    assert table.feature_matrix[1, 1] == pytest.approx(2_000_000.0)
    assert table.feature_matrix[1, 2] == pytest.approx(0.5)


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
        .otherwise(pl.col("priority_fee_p50"))
        .alias("priority_fee_p50")
    )

    with pytest.raises(ValueError, match="prev_priority_fee_p50"):
        _contract(["prev_priority_fee_p50"]).build_table(frame)


def test_feature_selection_validation_rejects_unknown_outputs() -> None:
    with pytest.raises(ValueError, match="Unknown feature outputs"):
        validate_feature_selection("core_fee_dynamics", ("raw_producer_address",))


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


def test_core_fee_dynamics_local_trends_config_adds_restored_safe_outputs() -> None:
    baseline = cast(dict[str, object], load_named_group("core_fee_dynamics", "features"))
    local_trends = cast(
        dict[str, object],
        load_named_group("core_fee_dynamics_local_trends", "features"),
    )
    baseline_outputs = cast(list[str], baseline["outputs"])
    local_outputs = cast(list[str], local_trends["outputs"])

    assert baseline_outputs == BASELINE_OUTPUTS
    assert local_trends["id"] == "core_fee_dynamics_local_trends"
    assert local_outputs[: len(baseline_outputs)] == baseline_outputs
    assert "elapsed_seconds" not in local_outputs
    assert "time_since_start" not in local_outputs
    assert "dlog_base_fee" in local_outputs
    assert "prev_gas_utilization_lag6" in local_outputs
    assert "roll200_std_prev_gas_utilization" in local_outputs
    coerce_features_config(local_trends)

    with pytest.raises(ValueError, match="Unknown feature outputs: dlog_base_fee"):
        coerce_features_config(
            {
                "id": "core_fee_dynamics",
                "outputs": [*baseline_outputs, "dlog_base_fee"],
            }
        )


def test_core_fee_dynamics_local_trends_builds_finite_aligned_feature_table() -> None:
    contract = _contract(
        [
            "dlog_base_fee",
            "base_fee_trend",
            "dlog_base_fee_lag6",
            "prev_gas_utilization_lag6",
            "roll200_mean_logfee",
            "roll200_std_prev_gas_utilization",
        ],
        features_id="core_fee_dynamics_local_trends",
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
        ],
        features_id="core_fee_dynamics_local_trends",
    ).build_table(_frame(12))

    assert table.feature_matrix[3, 0] == pytest.approx((15_000_000 + 2) / 30_000_000)
    assert table.feature_matrix[3, 1] == pytest.approx((15_000_000 + 1) / 30_000_000)
    assert table.feature_matrix[3, 3] == pytest.approx(table.feature_matrix[2, 2])

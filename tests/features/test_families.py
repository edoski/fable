from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spice.config import coerce_feature_set_config
from spice.config.registry import load_named_group
from spice.features import (
    FeaturePrerequisites,
    compile_feature_contract,
    validate_feature_selection,
)
from spice.features.core import feature_graph_fingerprint


def _block_frame() -> pl.DataFrame:
    row_count = 210
    return pl.DataFrame(
        {
            "block_number": np.arange(10_000, 10_000 + row_count, dtype=np.int64),
            "timestamp": np.arange(row_count, dtype=np.int64) * 12,
            "base_fee_per_gas": np.arange(1_000, 1_000 + row_count, dtype=np.int64),
            "gas_used": np.full(row_count, 18_000_000, dtype=np.int64),
            "gas_limit": np.full(row_count, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(row_count, dtype=np.int64),
        }
    ).sample(fraction=1.0, shuffle=True, seed=7)


def _time_frame() -> pl.DataFrame:
    row_count = 30
    return pl.DataFrame(
        {
            "block_number": np.arange(20_000, 20_000 + row_count, dtype=np.int64),
            "timestamp": np.arange(row_count, dtype=np.int64) * 30,
            "base_fee_per_gas": np.arange(2_000, 2_000 + row_count, dtype=np.int64),
            "gas_used": np.full(row_count, 21_000_000, dtype=np.int64),
            "gas_limit": np.full(row_count, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(row_count, dtype=np.int64),
        }
    ).sample(fraction=1.0, shuffle=True, seed=11)


def _block_open_frame() -> pl.DataFrame:
    row_count = 210
    return pl.DataFrame(
        {
            "block_number": np.arange(30_000, 30_000 + row_count, dtype=np.int64),
            "timestamp": np.arange(row_count, dtype=np.int64) * 3600,
            "base_fee_per_gas": np.arange(3_000, 3_000 + row_count, dtype=np.int64),
            "gas_used": np.arange(10_000_000, 10_000_000 + row_count, dtype=np.int64),
            "gas_limit": np.full(row_count, 30_000_000, dtype=np.int64),
            "chain_id": np.ones(row_count, dtype=np.int64),
        }
    ).sample(fraction=1.0, shuffle=True, seed=19)


def _build_contract(feature_set_id: str, family_id: str, outputs: list[str]):
    return compile_feature_contract(
        feature_set=coerce_feature_set_config(
            {
                "id": feature_set_id,
                "family": {"id": family_id},
                "outputs": outputs,
            }
        )
    )


def _assert_block_native(feature_contract, feature_table) -> None:
    assert feature_contract.feature_prerequisites == FeaturePrerequisites(
        history_seconds=0,
        warmup_rows=199,
    )
    np.testing.assert_array_equal(
        feature_table.feature_matrix[:5, 0],
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert np.isnan(feature_table.feature_matrix[8, 1])
    np.testing.assert_allclose(
        feature_table.feature_matrix[9, 1],
        np.log(np.arange(1_000, 1_010, dtype=np.float64)).mean(),
    )
    assert np.isnan(feature_table.feature_matrix[198, 2])
    assert feature_table.feature_matrix[199, 2] > 0.0


def _assert_time_native(feature_contract, feature_table) -> None:
    assert feature_contract.feature_prerequisites == FeaturePrerequisites(
        history_seconds=600,
        warmup_rows=0,
    )
    np.testing.assert_array_equal(
        feature_table.feature_matrix[:4, 0],
        np.array([0.0, 30.0, 30.0, 30.0], dtype=np.float32),
    )
    assert np.isnan(feature_table.feature_matrix[1, 1])
    np.testing.assert_allclose(
        feature_table.feature_matrix[2, 1],
        np.log(np.arange(2_000, 2_003, dtype=np.float64)).mean(),
    )
    assert np.isnan(feature_table.feature_matrix[19, 2])
    assert feature_table.feature_matrix[20, 2] > 0.0


def _assert_professor_block_native(feature_contract, feature_table) -> None:
    assert feature_contract.feature_prerequisites == FeaturePrerequisites(
        history_seconds=0,
        warmup_rows=9,
    )
    np.testing.assert_allclose(
        feature_table.feature_matrix[:3, 0],
        np.log1p(np.arange(1_000, 1_003, dtype=np.float64)),
    )
    np.testing.assert_array_equal(
        feature_table.feature_matrix[:4, 1],
        np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        feature_table.feature_matrix[9, 2],
        np.log1p(np.arange(1_000, 1_010, dtype=np.float64)).min(),
    )
    assert np.isnan(feature_table.feature_matrix[0, 3])
    assert feature_table.feature_matrix[1, 3] == pytest.approx(60.0)
    assert feature_table.feature_matrix[1, 4] == pytest.approx(1.0)
    assert feature_table.feature_matrix[199, 4] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        feature_table.feature_matrix[:4, 5],
        np.array([0.0, 12.0, 24.0, 36.0], dtype=np.float32),
    )
    assert np.isnan(feature_table.feature_matrix[8, 6])
    assert feature_table.feature_matrix[9, 6] == pytest.approx(0.0)


def _assert_professor_block_open(feature_contract, feature_table) -> None:
    assert feature_contract.feature_prerequisites == FeaturePrerequisites(
        history_seconds=0,
        warmup_rows=2,
    )
    np.testing.assert_allclose(
        feature_table.feature_matrix[:3, 0],
        np.log1p(np.arange(3_000, 3_003, dtype=np.float64)),
    )
    assert np.isnan(feature_table.feature_matrix[0, 1])
    expected_ratio0 = (10_000_000 / 30_000_000) * 100.0
    expected_ratio1 = (10_000_001 / 30_000_000) * 100.0
    assert feature_table.feature_matrix[1, 1] == pytest.approx(expected_ratio0)
    assert feature_table.feature_matrix[2, 1] == pytest.approx(expected_ratio1)
    assert np.isnan(feature_table.feature_matrix[0, 2])
    assert feature_table.feature_matrix[1, 2] == pytest.approx(3600.0)
    assert feature_table.feature_matrix[2, 2] == pytest.approx(3600.0)
    np.testing.assert_allclose(
        feature_table.feature_matrix[1:3, 3],
        np.array([0.0, np.sin(2.0 * np.pi / 24.0)], dtype=np.float32),
    )
    np.testing.assert_allclose(
        feature_table.feature_matrix[1:3, 4],
        np.array([0.0, 3600.0], dtype=np.float32),
    )
    current_dlog = np.log1p(3_002.0) - np.log1p(3_001.0)
    assert feature_table.feature_matrix[2, 5] == pytest.approx(current_dlog)
    previous_dlog = np.log1p(3_001.0) - np.log1p(3_000.0)
    assert feature_table.feature_matrix[2, 6] == pytest.approx(previous_dlog)


@pytest.mark.parametrize(
    ("feature_set_id", "family_id", "outputs", "frame_factory", "assertions"),
    [
        (
            "test_block_native",
            "block_native",
            [
                "elapsed_blocks",
                "rolling_mean_log_base_fee_10",
                "trend_slope_200",
            ],
            _block_frame,
            _assert_block_native,
        ),
        (
            "test_time_native",
            "time_native",
            [
                "seconds_since_previous_block",
                "rolling_mean_log_base_fee_60s",
                "trend_slope_600s",
            ],
            _time_frame,
            _assert_time_native,
        ),
        (
            "test_professor_block_native",
            "block_native",
            [
                "log_base_fee_per_gas",
                "dt_seconds",
                "roll10_min_logfee",
                "gas_ratio_lag1",
                "base_fee_trend",
                "time_since_start",
                "roll10_std_gr",
            ],
            _block_frame,
            _assert_professor_block_native,
        ),
        (
            "test_professor_block_open",
            "block_open_native",
            [
                "log_base_fee_per_gas",
                "gas_ratio",
                "dt_seconds",
                "hour_sin",
                "time_since_start",
                "dlog_base_fee",
                "dlogfee_lag1",
            ],
            _block_open_frame,
            _assert_professor_block_open,
        ),
    ],
)
def test_feature_family_builds_expected_table(
    feature_set_id: str,
    family_id: str,
    outputs: list[str],
    frame_factory,
    assertions,
) -> None:
    feature_contract = _build_contract(feature_set_id, family_id, outputs)
    feature_table = feature_contract.build_table(frame_factory())
    assert feature_table.feature_names == tuple(outputs)
    assertions(feature_contract, feature_table)


@pytest.mark.parametrize(
    ("feature_set_id", "feature_family_id", "feature_names", "message"),
    [
        ("test", "block_native", (), "feature_set.outputs must not be empty"),
        (
            "test",
            "block_native",
            ("elapsed_blocks", "elapsed_blocks"),
            "feature_set.outputs must not contain duplicates: elapsed_blocks",
        ),
        (
            "test",
            "block_native",
            ("elapsed_blocks", "missing"),
            "Unknown feature outputs: missing",
        ),
    ],
)
def test_validate_feature_selection_rejects_invalid_requests(
    feature_set_id: str,
    feature_family_id: str,
    feature_names: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_feature_selection(feature_set_id, feature_family_id, feature_names)


def test_feature_fingerprint_changes_when_source_bytes_change(tmp_path: Path) -> None:
    primary = tmp_path / "primary.py"
    helper = tmp_path / "helper.py"
    primary.write_text("value = 1\n", encoding="utf-8")
    helper.write_text("value = 2\n", encoding="utf-8")

    first = feature_graph_fingerprint(
        "block_native",
        ("elapsed_blocks", "trend_slope_200"),
        fingerprint_sources=(primary, helper),
    )

    helper.write_text("value = 3\n", encoding="utf-8")

    second = feature_graph_fingerprint(
        "block_native",
        ("elapsed_blocks", "trend_slope_200"),
        fingerprint_sources=(primary, helper),
    )

    assert first != second


def test_feature_family_requires_declared_block_columns() -> None:
    feature_contract = _build_contract(
        "test_missing_columns",
        "block_native",
        ["gas_ratio", "dt_seconds"],
    )
    frame = _block_frame().drop("gas_limit")

    with pytest.raises(ValueError, match="missing block columns: gas_limit"):
        feature_contract.build_table(frame)


def test_block_open_gas_ratio_rolls_account_for_lagged_warmup() -> None:
    feature_contract = _build_contract(
        "test_block_open_rolls",
        "block_open_native",
        ["gas_ratio", "roll10_std_gr"],
    )
    feature_table = feature_contract.build_table(_block_open_frame())

    assert feature_contract.feature_prerequisites == FeaturePrerequisites(
        history_seconds=0,
        warmup_rows=10,
    )
    assert np.isnan(feature_table.feature_matrix[9, 1])
    assert np.isfinite(feature_table.feature_matrix[10, 1])


@pytest.mark.parametrize(
    ("feature_set_name", "family_id", "present", "absent"),
    [
        (
            "icdcs_2026_professor_no_time_since_start",
            "block_native",
            {"dt_seconds", "hour_sin", "dow_cos"},
            {"time_since_start"},
        ),
        (
            "icdcs_2026_professor_no_time_features",
            "block_native",
            {"dt_seconds"},
            {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "time_since_start"},
        ),
        (
            "icdcs_2026_professor_calendar_only_time",
            "block_native",
            {"hour_sin", "hour_cos", "dow_sin", "dow_cos"},
            {"dt_seconds", "time_since_start"},
        ),
        (
            "icdcs_2026_professor_block_open_no_time_since_start",
            "block_open_native",
            {"dt_seconds", "hour_sin", "dow_cos"},
            {"time_since_start"},
        ),
        (
            "icdcs_2026_professor_block_open_no_time_features",
            "block_open_native",
            {"dt_seconds"},
            {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "time_since_start"},
        ),
        (
            "icdcs_2026_professor_block_open_calendar_only_time",
            "block_open_native",
            {"hour_sin", "hour_cos", "dow_sin", "dow_cos"},
            {"dt_seconds", "time_since_start"},
        ),
    ],
)
def test_named_time_ablation_feature_sets_resolve_as_intended(
    feature_set_name: str,
    family_id: str,
    present: set[str],
    absent: set[str],
) -> None:
    feature_set = coerce_feature_set_config(load_named_group(feature_set_name, "feature_set"))

    assert feature_set.id == feature_set_name
    assert feature_set.family.id == family_id
    assert present.issubset(set(feature_set.outputs))
    assert absent.isdisjoint(set(feature_set.outputs))

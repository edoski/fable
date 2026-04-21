from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spice.config import coerce_feature_set_config
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
        warmup_rows=199,
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
    assert feature_table.feature_matrix[1, 3] == pytest.approx(0.6)
    assert np.isnan(feature_table.feature_matrix[198, 4])
    assert np.isfinite(feature_table.feature_matrix[199, 4])


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
                "seconds_since_previous_block",
                "rolling_min_log_base_fee_per_gas_10",
                "gas_utilization_lag1",
                "trend_slope_log_base_fee_per_gas_200",
            ],
            _block_frame,
            _assert_professor_block_native,
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

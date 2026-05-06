from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from spice.corpus.validation import (
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)
from tests.dataset_helpers import make_block_rows


def test_validate_contiguous_block_frame_reports_shape_errors() -> None:
    rows = make_block_rows(
        4,
        start_block=100,
        start_timestamp=1_000,
        chain_id=2,
    )
    rows[2]["block_number"] = rows[1]["block_number"]
    rows[3]["block_number"] = cast(int, rows[2]["block_number"]) + 2
    report = validate_contiguous_block_frame(
        pl.DataFrame(rows),
        dataset_path=Path("/tmp/history"),
        expected_chain_id=1,
    )

    assert report.status == "error"
    assert report.duplicate_count == 1
    assert report.gap_count == 1
    assert any("chain_id mismatch" in error for error in report.errors)
    assert any("duplicate" in error for error in report.errors)
    assert any("gap" in error for error in report.errors)


def test_validate_exact_window_frame_reports_out_of_range_rows() -> None:
    rows = make_block_rows(
        3,
        start_block=200,
        start_timestamp=2_000,
        chain_id=1,
    )
    rows[0]["timestamp"] = 1_990
    rows[2]["timestamp"] = 2_030
    report = validate_exact_window_frame(
        pl.DataFrame(rows),
        dataset_path=Path("/tmp/evaluation"),
        expected_chain_id=1,
        expected_start_timestamp=2_000,
        expected_end_timestamp=2_024,
    )

    assert report.status == "error"
    assert report.below_start_count == 1
    assert report.above_end_count == 1
    assert any("out-of-range timestamps" in error for error in report.errors)


def test_validate_contiguous_block_frame_requires_source_facts() -> None:
    rows = make_block_rows(
        3,
        start_block=300,
        start_timestamp=3_000,
        chain_id=1,
    )
    rows[1]["priority_fee_p50"] = None

    report = validate_contiguous_block_frame(
        pl.DataFrame(rows),
        dataset_path=Path("/tmp/history"),
        expected_chain_id=1,
        required_columns=frozenset({"priority_fee_p50"}),
    )

    assert report.status == "error"
    assert any("null required source columns: priority_fee_p50" in error for error in report.errors)

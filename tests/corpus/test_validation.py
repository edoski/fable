from __future__ import annotations

from pathlib import Path

import polars as pl

from spice.corpus.validation import (
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)


def test_validate_contiguous_block_frame_reports_shape_errors(make_block_rows) -> None:
    rows = make_block_rows(
        4,
        start_block=100,
        start_timestamp=1_000,
        chain_id=2,
    )
    rows[2]["block_number"] = rows[1]["block_number"]
    rows[3]["block_number"] = rows[2]["block_number"] + 2
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


def test_validate_exact_window_frame_reports_out_of_range_rows(make_block_rows) -> None:
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

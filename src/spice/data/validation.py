"""Shared exact-window validation for block datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from .io import read_block_dataset

ValidationStatus = Literal["clean", "warning", "error"]
EXACT_WINDOW_COLUMNS = ("block_number", "timestamp", "chain_id")


class ValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockDatasetValidationReport(ValidationModel):
    dataset_path: Path
    expected_start_timestamp: int
    expected_end_timestamp: int
    row_count: int = 0
    first_block_number: int | None = None
    last_block_number: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    chain_id: int | None = None
    duplicate_count: int = 0
    gap_count: int = 0
    below_start_count: int = 0
    above_end_count: int = 0
    status: ValidationStatus = "clean"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _finalize_status(report: BlockDatasetValidationReport) -> None:
    if report.errors:
        report.status = "error"
    elif report.warnings:
        report.status = "warning"
    else:
        report.status = "clean"


def _coerce_exact_window_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in EXACT_WINDOW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Block dataset is missing required validation columns: " + ", ".join(missing)
        )
    return frame.select(
        [
            pl.col("block_number").cast(pl.Int64, strict=True),
            pl.col("timestamp").cast(pl.Int64, strict=True),
            pl.col("chain_id").cast(pl.Int64, strict=True),
        ]
    ).sort("block_number")


def validate_exact_window_frame(
    frame: pl.DataFrame,
    *,
    dataset_path: Path,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> BlockDatasetValidationReport:
    report = BlockDatasetValidationReport(
        dataset_path=dataset_path,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    try:
        exact_frame = _coerce_exact_window_frame(frame)
    except Exception as exc:
        report.errors.append(str(exc))
        _finalize_status(report)
        return report

    if exact_frame.height == 0:
        report.errors.append("Block dataset is empty")
        _finalize_status(report)
        return report

    block_numbers = exact_frame["block_number"].to_list()
    timestamps = exact_frame["timestamp"].to_list()
    chain_ids = exact_frame["chain_id"].to_list()

    report.row_count = len(block_numbers)
    report.first_block_number = int(block_numbers[0])
    report.last_block_number = int(block_numbers[-1])
    report.first_timestamp = int(timestamps[0])
    report.last_timestamp = int(timestamps[-1])

    unique_chain_ids = sorted(set(chain_ids))
    if len(unique_chain_ids) != 1:
        report.errors.append("Block dataset must contain exactly one chain_id")
    else:
        report.chain_id = int(unique_chain_ids[0])
        if report.chain_id != expected_chain_id:
            report.errors.append(
                f"Block dataset chain_id mismatch: expected {expected_chain_id}, got "
                f"{report.chain_id}"
            )

    for left, right in zip(block_numbers, block_numbers[1:], strict=False):
        if right == left:
            report.duplicate_count += 1
        elif right != left + 1:
            report.gap_count += 1

    report.below_start_count = sum(
        1 for timestamp in timestamps if timestamp < expected_start_timestamp
    )
    report.above_end_count = sum(
        1 for timestamp in timestamps if timestamp >= expected_end_timestamp
    )

    if report.duplicate_count:
        report.errors.append(
            f"Detected {report.duplicate_count} duplicate block_number transition(s)"
        )
    if report.gap_count:
        report.errors.append(f"Detected {report.gap_count} block-number gap(s)")
    if report.below_start_count or report.above_end_count:
        report.errors.append(
            f"Detected out-of-range timestamps: below_start={report.below_start_count}, "
            f"above_end={report.above_end_count}"
        )

    _finalize_status(report)
    return report


def validate_exact_window_dataset(
    dataset_path: Path,
    *,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> BlockDatasetValidationReport:
    report = BlockDatasetValidationReport(
        dataset_path=dataset_path,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    try:
        frame = read_block_dataset(dataset_path, columns=EXACT_WINDOW_COLUMNS)
    except Exception as exc:
        report.errors.append(str(exc))
        _finalize_status(report)
        return report

    return validate_exact_window_frame(
        frame,
        dataset_path=dataset_path,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )

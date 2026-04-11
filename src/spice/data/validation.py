"""Shared exact-window validation for block datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import Field

from .io import read_block_dataset
from .validation_base import ValidationModel, ValidationStatus, finalize_validation_status

EXACT_WINDOW_COLUMNS = ("block_number", "timestamp", "chain_id")


@dataclass(slots=True)
class ExactWindowSummary:
    row_count: int
    first_block_number: int
    last_block_number: int
    first_timestamp: int
    last_timestamp: int
    chain_ids: tuple[int, ...]
    duplicate_count: int
    gap_count: int
    below_start_count: int
    above_end_count: int


@dataclass(slots=True)
class ExactWindowAssessment:
    chain_id: int | None
    errors: tuple[str, ...]


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
    errors: list[str] = Field(default_factory=list)


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


def summarize_exact_window_frame(
    frame: pl.DataFrame,
    *,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> ExactWindowSummary:
    exact_frame = _coerce_exact_window_frame(frame)
    if exact_frame.height == 0:
        raise ValueError("Block dataset is empty")

    block_numbers = [int(value) for value in exact_frame["block_number"].to_list()]
    timestamps = [int(value) for value in exact_frame["timestamp"].to_list()]
    chain_ids = tuple(sorted({int(value) for value in exact_frame["chain_id"].to_list()}))

    duplicate_count = 0
    gap_count = 0
    for left, right in zip(block_numbers, block_numbers[1:], strict=False):
        if right == left:
            duplicate_count += 1
        elif right != left + 1:
            gap_count += 1

    below_start_count = sum(
        1 for timestamp in timestamps if timestamp < expected_start_timestamp
    )
    above_end_count = sum(
        1 for timestamp in timestamps if timestamp >= expected_end_timestamp
    )
    return ExactWindowSummary(
        row_count=len(block_numbers),
        first_block_number=block_numbers[0],
        last_block_number=block_numbers[-1],
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
        chain_ids=chain_ids,
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        below_start_count=below_start_count,
        above_end_count=above_end_count,
    )


def summarize_exact_window_dataset(
    dataset_path: Path,
    *,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> ExactWindowSummary:
    frame = read_block_dataset(dataset_path, columns=EXACT_WINDOW_COLUMNS)
    return summarize_exact_window_frame(
        frame,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )


def assess_exact_window_summary(
    summary: ExactWindowSummary,
    *,
    expected_chain_id: int,
) -> ExactWindowAssessment:
    errors: list[str] = []
    chain_id: int | None = None
    if len(summary.chain_ids) != 1:
        errors.append("Block dataset must contain exactly one chain_id")
    else:
        chain_id = summary.chain_ids[0]
        if chain_id != expected_chain_id:
            errors.append(
                f"Block dataset chain_id mismatch: expected {expected_chain_id}, got {chain_id}"
            )

    if summary.duplicate_count:
        errors.append(
            f"Detected {summary.duplicate_count} duplicate block_number transition(s)"
        )
    if summary.gap_count:
        errors.append(f"Detected {summary.gap_count} block-number gap(s)")
    if summary.below_start_count or summary.above_end_count:
        errors.append(
            f"Detected out-of-range timestamps: below_start={summary.below_start_count}, "
            f"above_end={summary.above_end_count}"
        )
    return ExactWindowAssessment(chain_id=chain_id, errors=tuple(errors))


def _apply_exact_window_summary(
    report: BlockDatasetValidationReport,
    summary: ExactWindowSummary,
    *,
    expected_chain_id: int,
) -> None:
    report.row_count = summary.row_count
    report.first_block_number = summary.first_block_number
    report.last_block_number = summary.last_block_number
    report.first_timestamp = summary.first_timestamp
    report.last_timestamp = summary.last_timestamp
    report.duplicate_count = summary.duplicate_count
    report.gap_count = summary.gap_count
    report.below_start_count = summary.below_start_count
    report.above_end_count = summary.above_end_count

    assessment = assess_exact_window_summary(
        summary,
        expected_chain_id=expected_chain_id,
    )
    report.chain_id = assessment.chain_id
    report.errors.extend(assessment.errors)


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
        summary = summarize_exact_window_frame(
            frame,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
        )
    except Exception as exc:
        report.errors.append(str(exc))
        finalize_validation_status(report)
        return report

    _apply_exact_window_summary(
        report,
        summary,
        expected_chain_id=expected_chain_id,
    )
    finalize_validation_status(report)
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
        summary = summarize_exact_window_dataset(
            dataset_path,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
        )
    except Exception as exc:
        report.errors.append(str(exc))
        finalize_validation_status(report)
        return report

    _apply_exact_window_summary(
        report,
        summary,
        expected_chain_id=expected_chain_id,
    )
    finalize_validation_status(report)
    return report

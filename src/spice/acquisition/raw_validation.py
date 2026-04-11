"""Validation helpers for raw cryo block pulls."""

from __future__ import annotations

import re
from pathlib import Path

import pandera.polars as pa
import polars as pl
from pydantic import Field

from ..data.io import iter_block_files
from ..data.validation import assess_exact_window_summary, summarize_exact_window_dataset
from ..data.validation_base import ValidationModel, ValidationStatus, finalize_validation_status

RAW_BLOCK_FILENAME_RE = re.compile(
    r"^(?P<chain>[a-z0-9_]+)__blocks__(?P<start>\d+)_to_(?P<end>\d+)$"
)


class RawFileSummary(ValidationModel):
    path: Path
    filename_chain: str
    range_start: int
    range_end: int
    row_count: int
    first_block_number: int
    last_block_number: int
    first_timestamp: int
    last_timestamp: int
    chain_id_mismatch_count: int
    duplicate_count: int
    non_sequential_transition_count: int
    below_start_count: int
    above_end_count: int
    internal_timestamp_violation: bool


class RawPullValidationReport(ValidationModel):
    dataset_path: Path
    expected_start_timestamp: int
    expected_end_timestamp: int
    file_count: int = 0
    row_count: int = 0
    first_block_number: int | None = None
    last_block_number: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    gap_count: int = 0
    overlap_count: int = 0
    duplicate_count: int = 0
    chain_id_mismatch_count: int = 0
    below_start_count: int = 0
    above_end_count: int = 0
    status: ValidationStatus = "clean"
    errors: list[str] = Field(default_factory=list)


def _read_lightweight_columns(path: Path) -> tuple[list[int], list[int], list[int]]:
    frame = pl.read_parquet(path, columns=["block_number", "timestamp", "chain_id"])
    return (
        frame["block_number"].cast(pl.Int64).to_list(),
        frame["timestamp"].cast(pl.Int64).to_list(),
        frame["chain_id"].cast(pl.Int64).to_list(),
    )


def _scan_timestamp_window(
    timestamps: list[int],
    *,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> tuple[int, int, bool]:
    below_start_count = 0
    above_end_count = 0
    internal_timestamp_violation = False
    for index, timestamp in enumerate(timestamps):
        if timestamp < expected_start_timestamp:
            below_start_count += 1
            if index not in (0, len(timestamps) - 1):
                internal_timestamp_violation = True
        elif timestamp >= expected_end_timestamp:
            above_end_count += 1
            if index not in (0, len(timestamps) - 1):
                internal_timestamp_violation = True
    return below_start_count, above_end_count, internal_timestamp_violation


def _scan_block_transitions(block_numbers: list[int]) -> tuple[int, int]:
    duplicate_count = 0
    non_sequential_transition_count = 0
    for left, right in zip(block_numbers, block_numbers[1:], strict=False):
        if right == left:
            duplicate_count += 1
        elif right != left + 1:
            non_sequential_transition_count += 1
    return duplicate_count, non_sequential_transition_count


def _summarize_file(
    path: Path,
    *,
    expected_chain_name: str,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> RawFileSummary:
    match = RAW_BLOCK_FILENAME_RE.match(path.stem)
    if match is None:
        raise ValueError(
            f"Malformed raw block filename: {path.name}. "
            "Expected <chain>__blocks__<start>_to_<end>."
        )
    filename_chain = match.group("chain")
    range_start = int(match.group("start"))
    range_end = int(match.group("end"))
    if filename_chain != expected_chain_name:
        raise ValueError(
            f"Raw block filename chain mismatch for {path.name}: "
            f"expected {expected_chain_name}, got {filename_chain}"
        )
    if range_end < range_start:
        raise ValueError(f"Invalid raw block filename range in {path.name}")

    block_numbers, timestamps, chain_ids = _read_lightweight_columns(path)
    if not block_numbers:
        raise ValueError(f"Raw block file is empty: {path.name}")
    below_start_count, above_end_count, internal_timestamp_violation = _scan_timestamp_window(
        timestamps,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    duplicate_count, non_sequential_transition_count = _scan_block_transitions(block_numbers)
    return RawFileSummary(
        path=path,
        filename_chain=filename_chain,
        range_start=range_start,
        range_end=range_end,
        row_count=len(block_numbers),
        first_block_number=block_numbers[0],
        last_block_number=block_numbers[-1],
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
        chain_id_mismatch_count=sum(1 for chain_id in chain_ids if chain_id != expected_chain_id),
        duplicate_count=duplicate_count,
        non_sequential_transition_count=non_sequential_transition_count,
        below_start_count=below_start_count,
        above_end_count=above_end_count,
        internal_timestamp_violation=internal_timestamp_violation,
    )


def _summary_frame(summaries: list[RawFileSummary]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    previous_range_end: int | None = None
    previous_last_block: int | None = None
    for summary in summaries:
        gap_after_previous = 0
        overlap_with_previous = 0
        duplicate_transition_from_previous = 0
        block_order_violation = False
        if previous_range_end is not None:
            if summary.range_start > previous_range_end + 1:
                gap_after_previous = summary.range_start - previous_range_end - 1
            elif summary.range_start <= previous_range_end:
                overlap_with_previous = previous_range_end - summary.range_start + 1
        if previous_last_block is not None:
            if summary.first_block_number == previous_last_block:
                duplicate_transition_from_previous = 1
            elif summary.first_block_number < previous_last_block:
                block_order_violation = True
        rows.append(
            {
                **summary.model_dump(mode="json"),
                "path": str(summary.path),
                "expected_row_count": summary.range_end - summary.range_start + 1,
                "gap_after_previous": gap_after_previous,
                "overlap_with_previous": overlap_with_previous,
                "duplicate_transition_from_previous": duplicate_transition_from_previous,
                "block_order_violation": block_order_violation,
            }
        )
        previous_range_end = summary.range_end
        previous_last_block = summary.last_block_number
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _build_summary_schema(chunk_size: int) -> pa.DataFrameSchema:
    def _frame(data: object) -> pl.DataFrame:
        return data.lazyframe.collect()  # type: ignore[union-attr]

    return pa.DataFrameSchema(
        {
            "path": pa.Column(str, nullable=False, unique=True),
            "filename_chain": pa.Column(str, nullable=False),
            "range_start": pa.Column(pl.Int64, nullable=False),
            "range_end": pa.Column(pl.Int64, nullable=False),
            "row_count": pa.Column(pl.Int64, nullable=False),
            "first_block_number": pa.Column(pl.Int64, nullable=False),
            "last_block_number": pa.Column(pl.Int64, nullable=False),
            "first_timestamp": pa.Column(pl.Int64, nullable=False),
            "last_timestamp": pa.Column(pl.Int64, nullable=False),
            "chain_id_mismatch_count": pa.Column(pl.Int64, nullable=False),
            "duplicate_count": pa.Column(pl.Int64, nullable=False),
            "non_sequential_transition_count": pa.Column(pl.Int64, nullable=False),
            "below_start_count": pa.Column(pl.Int64, nullable=False),
            "above_end_count": pa.Column(pl.Int64, nullable=False),
            "internal_timestamp_violation": pa.Column(pl.Boolean, nullable=False),
            "expected_row_count": pa.Column(pl.Int64, nullable=False),
            "gap_after_previous": pa.Column(pl.Int64, nullable=False),
            "overlap_with_previous": pa.Column(pl.Int64, nullable=False),
            "duplicate_transition_from_previous": pa.Column(pl.Int64, nullable=False),
            "block_order_violation": pa.Column(pl.Boolean, nullable=False),
        },
        strict=True,
        checks=[
            pa.Check(
                lambda data: (_frame(data)["range_end"] >= _frame(data)["range_start"]).all(),
                error="Invalid raw block filename range(s)",
            ),
            pa.Check(
                lambda data: (
                    _frame(data)["row_count"] == _frame(data)["expected_row_count"]
                ).all(),
                error="row_count does not match filename range size",
            ),
            pa.Check(
                lambda data: (_frame(data)["row_count"] <= chunk_size).all(),
                error="row_count exceeds configured chunk_size",
            ),
            pa.Check(
                lambda data: (
                    _frame(data)["first_block_number"] == _frame(data)["range_start"]
                ).all(),
                error="first block_number does not match filename start",
            ),
            pa.Check(
                lambda data: (_frame(data)["last_block_number"] == _frame(data)["range_end"]).all(),
                error="last block_number does not match filename end",
            ),
            pa.Check(
                lambda data: (_frame(data)["non_sequential_transition_count"] == 0).all(),
                error="non-sequential block transitions detected inside a file",
            ),
            pa.Check(
                lambda data: (_frame(data)["chain_id_mismatch_count"] == 0).all(),
                error="chain_id mismatch detected",
            ),
            pa.Check(
                lambda data: (_frame(data)["gap_after_previous"] == 0).all(),
                error="cross-file block-range gaps detected",
            ),
            pa.Check(
                lambda data: (_frame(data)["overlap_with_previous"] == 0).all(),
                error="cross-file block-range overlaps detected",
            ),
            pa.Check(
                lambda data: (_frame(data)["duplicate_transition_from_previous"] == 0).all(),
                error="duplicate cross-file block transitions detected",
            ),
            pa.Check(
                lambda data: (~_frame(data)["block_order_violation"]).all(),
                error="cross-file block ordering violation detected",
            ),
        ],
    )


def _merge_summary_into_report(report: RawPullValidationReport, summary: RawFileSummary) -> None:
    report.row_count += summary.row_count
    report.chain_id_mismatch_count += summary.chain_id_mismatch_count
    if report.first_block_number is None:
        report.first_block_number = summary.first_block_number
        report.first_timestamp = summary.first_timestamp
    report.last_block_number = summary.last_block_number
    report.last_timestamp = summary.last_timestamp


def _apply_exact_window_summary(
    report: RawPullValidationReport,
    *,
    expected_chain_id: int,
) -> None:
    summary = summarize_exact_window_dataset(
        report.dataset_path,
        expected_start_timestamp=report.expected_start_timestamp,
        expected_end_timestamp=report.expected_end_timestamp,
    )
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
    for error in assessment.errors:
        if error not in report.errors:
            report.errors.append(error)


def validate_raw_pull(
    dataset_path: Path,
    *,
    expected_chain_name: str,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
    expected_chunk_size: int,
) -> RawPullValidationReport:
    report = RawPullValidationReport(
        dataset_path=dataset_path,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    summaries: list[RawFileSummary] = []
    try:
        paths = iter_block_files(dataset_path)
    except ValueError as exc:
        report.errors.append(str(exc))
        finalize_validation_status(report)
        return report

    for path in paths:
        try:
            summaries.append(
                _summarize_file(
                    path,
                    expected_chain_name=expected_chain_name,
                    expected_chain_id=expected_chain_id,
                    expected_start_timestamp=expected_start_timestamp,
                    expected_end_timestamp=expected_end_timestamp,
                )
            )
        except ValueError as exc:
            report.errors.append(str(exc))
    summaries.sort(key=lambda item: (item.range_start, item.range_end, str(item.path)))
    report.file_count = len(summaries)

    summary_frame = _summary_frame(summaries)
    if summary_frame.height:
        try:
            _build_summary_schema(expected_chunk_size).validate(summary_frame)
        except Exception as exc:
            report.errors.append(str(exc))

    for summary in summaries:
        _merge_summary_into_report(report, summary)

    if summary_frame.height:
        file_range_gap_count = int(summary_frame["gap_after_previous"].gt(0).sum())
        report.overlap_count = int(summary_frame["overlap_with_previous"].gt(0).sum())
        if file_range_gap_count:
            report.errors.append(
                f"Detected {file_range_gap_count} block-range gap(s) across files"
            )
        if report.overlap_count:
            report.errors.append(f"Detected {report.overlap_count} overlapping file-range pair(s)")
        if int(summary_frame["block_order_violation"].sum()):
            report.errors.append("Detected cross-file block ordering violation(s)")

        if int(summary_frame["non_sequential_transition_count"].sum()):
            report.errors.append("Detected non-sequential block transition(s) inside files")
        if report.chain_id_mismatch_count:
            report.errors.append(
                f"Detected {report.chain_id_mismatch_count} row(s) with chain_id different "
                f"from {expected_chain_id}"
            )

    try:
        _apply_exact_window_summary(
            report,
            expected_chain_id=expected_chain_id,
        )
    except Exception as exc:
        report.errors.append(str(exc))
    finalize_validation_status(report)
    return report

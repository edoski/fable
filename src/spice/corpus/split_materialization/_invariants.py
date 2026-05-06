"""Corpus split target validation invariants."""

from __future__ import annotations

from pathlib import Path

from ...acquisition import BlockPullPlan, BlockRange, TimestampRange
from ..io import load_block_frame
from ..validation import BlockDatasetValidationReport, validate_exact_window_frame
from ._chunks import filter_block_range
from ._policy import SplitDatasetCandidate


def validate_history_result(
    validation: BlockDatasetValidationReport,
    *,
    history_plan: BlockPullPlan,
) -> None:
    if validation.status != "clean":
        raise ValueError(f"Canonical history dataset validation failed: {validation}")
    if validation.last_block_number != history_plan.block_range.end - 1:
        raise ValueError(
            "History dataset does not end at the requested evaluation boundary: "
            f"expected last block {history_plan.block_range.end - 1}, "
            f"got {validation.last_block_number}"
        )
    if validation.first_block_number is None:
        raise ValueError("History dataset validation did not produce a first block number")
    if validation.first_block_number > history_plan.block_range.start:
        raise ValueError(
            "History dataset does not cover the requested oldest history block: "
            f"expected at most {history_plan.block_range.start}, "
            f"got {validation.first_block_number}"
        )


def validate_evaluation_result(
    validation: BlockDatasetValidationReport,
    *,
    evaluation_dir: Path,
    evaluation_plan: BlockPullPlan,
    expected_chain_id: int,
    required_columns: frozenset[str],
) -> None:
    exact_validation = validate_exact_window_frame(
        load_block_frame(evaluation_dir),
        dataset_path=evaluation_dir,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=evaluation_plan.window.start,
        expected_end_timestamp=evaluation_plan.window.end,
        required_columns=required_columns,
    )
    if exact_validation.status != "clean":
        raise ValueError(f"Canonical evaluation dataset validation failed: {exact_validation}")
    if exact_validation.first_block_number != evaluation_plan.block_range.start:
        raise ValueError(
            "Evaluation dataset does not start at the requested block boundary: "
            f"expected first block {evaluation_plan.block_range.start}, "
            f"got {exact_validation.first_block_number}"
        )
    if exact_validation.last_block_number != evaluation_plan.block_range.end - 1:
        raise ValueError(
            "Evaluation dataset does not end at the requested block boundary: "
            f"expected last block {evaluation_plan.block_range.end - 1}, "
            f"got {exact_validation.last_block_number}"
        )
    validation.status = exact_validation.status
    validation.below_start_count = exact_validation.below_start_count
    validation.above_end_count = exact_validation.above_end_count
    validation.expected_start_timestamp = exact_validation.expected_start_timestamp
    validation.expected_end_timestamp = exact_validation.expected_end_timestamp
    validation.errors = list(exact_validation.errors)


def reusable_range_matches_target_window(
    candidate: SplitDatasetCandidate,
    block_range: BlockRange,
    window: TimestampRange,
) -> bool:
    frame = filter_block_range(load_block_frame(candidate.path), block_range)
    if frame.height == 0:
        return False
    timestamps: list[int] = []
    for value in frame["timestamp"].to_list():
        if type(value) is not int:
            return False
        timestamps.append(value)
    return min(timestamps) >= window.start and max(timestamps) < window.end

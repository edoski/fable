"""Shared dataset build helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from ...acquisition.rpc import (
    BlockPullPlan,
    BlockRange,
    RpcController,
    TimestampRange,
    Web3BlockClient,
    pull_block_range,
)
from ...core.reporting import Reporter
from ...corpus.io import iter_block_files, load_block_frame, write_block_file
from ...corpus.validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)
from ..metadata import has_block_files


@dataclass(slots=True)
class ExistingDatasetState:
    path: Path
    frame: pl.DataFrame | None
    validation: BlockDatasetValidationReport
    file_count: int


class DatasetBuildOutcome(StrEnum):
    CREATED = "created"
    REUSED = "reused"
    EXTENDED = "extended"
    REBUILT = "rebuilt"


@dataclass(slots=True)
class DatasetBuildResult:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int
    promote_dir: Path | None
    pulled_blocks: bool
    outcome: DatasetBuildOutcome


StageUpdateCallback = Callable[[str, str | None], None]


def noop_stage_update(status: str, message: str | None = None) -> None:
    del status, message


def validate_block_dataset(
    path: Path,
    *,
    expected_chain_id: int,
) -> BlockDatasetValidationReport:
    try:
        frame = load_block_frame(path)
    except Exception as exc:  # pragma: no cover - surfaced in workflow smoke tests
        return BlockDatasetValidationReport(
            dataset_path=path,
            status="error",
            errors=[str(exc)],
        )
    return validate_contiguous_block_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
    )


def load_existing_dataset(
    path: Path,
    *,
    expected_chain_id: int,
) -> ExistingDatasetState | None:
    if not has_block_files(path):
        return None
    try:
        frame = load_block_frame(path)
    except Exception as exc:
        return ExistingDatasetState(
            path=path,
            frame=None,
            validation=BlockDatasetValidationReport(
                dataset_path=path,
                status="error",
                errors=[str(exc)],
            ),
            file_count=len(iter_block_files(path)),
        )
    validation = validate_contiguous_block_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
    )
    return ExistingDatasetState(
        path=path,
        frame=frame,
        validation=validation,
        file_count=len(iter_block_files(path)),
    )


def block_range_start(validation: BlockDatasetValidationReport) -> int:
    if validation.first_block_number is None:
        raise ValueError("validated dataset is missing the first block number")
    return validation.first_block_number


def block_range_end(validation: BlockDatasetValidationReport) -> int:
    if validation.last_block_number is None:
        raise ValueError("validated dataset is missing the last block number")
    return validation.last_block_number + 1


async def pull_plan_to_frame(
    *,
    block_client: Web3BlockClient,
    plan: BlockPullPlan,
    output_dir: Path,
    chunk_size: int,
    rpc_controller: RpcController,
    reporter: Reporter,
    progress_total: int | None = None,
    progress_completed: int = 0,
) -> pl.DataFrame:
    await pull_block_range(
        block_client,
        output_dir,
        plan=plan,
        chunk_size=chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
        progress_total=progress_total,
        progress_completed=progress_completed,
    )
    return load_block_frame(output_dir)


def filter_block_range(frame: pl.DataFrame, block_range: BlockRange) -> pl.DataFrame:
    return frame.filter(
        (pl.col("block_number") >= block_range.start)
        & (pl.col("block_number") < block_range.end)
    ).sort("block_number")


def write_block_dataset_dir(
    output_dir: Path,
    *,
    frame: pl.DataFrame,
    chunk_size: int,
    chain_name: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_frame = frame.sort("block_number")
    file_count = 0
    for start_index in range(0, sorted_frame.height, chunk_size):
        chunk = sorted_frame.slice(start_index, min(chunk_size, sorted_frame.height - start_index))
        start_block = int(chunk["block_number"][0])
        end_block = int(chunk["block_number"][-1])
        write_block_file(
            output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet",
            chunk,
        )
        file_count += 1
    return file_count


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
) -> None:
    exact_validation = validate_exact_window_frame(
        load_block_frame(evaluation_dir),
        dataset_path=evaluation_dir,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=evaluation_plan.window.start,
        expected_end_timestamp=evaluation_plan.window.end,
    )
    if exact_validation.status != "clean":
        raise ValueError(f"Canonical evaluation dataset validation failed: {exact_validation}")
    validation.status = exact_validation.status
    validation.below_start_count = exact_validation.below_start_count
    validation.above_end_count = exact_validation.above_end_count
    validation.expected_start_timestamp = exact_validation.expected_start_timestamp
    validation.expected_end_timestamp = exact_validation.expected_end_timestamp
    validation.errors = list(exact_validation.errors)


def combined_frame(*frames: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [frame for frame in frames if frame.height > 0],
        how="vertical",
    ).sort("block_number")


def partial_plan(
    block_client: Web3BlockClient,
    *,
    start_block: int,
    end_block: int,
    window: TimestampRange,
    chunk_size: int,
) -> BlockPullPlan | None:
    if end_block <= start_block:
        return None
    return block_client.plan_block_range(
        BlockRange(start=start_block, end=end_block),
        window=window,
        chunk_size=chunk_size,
    )

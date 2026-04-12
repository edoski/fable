"""Canonical dataset builders for acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..core.config import ExperimentConfig
from ..core.console import Reporter
from ..data.io import iter_block_files, load_block_frame, write_block_file
from ..data.validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)
from .metadata import has_block_files
from .rpc import BlockPullPlan, BlockRange, TimestampRange, Web3BlockClient


@dataclass(slots=True)
class ExistingDatasetState:
    path: Path
    frame: pl.DataFrame | None
    validation: BlockDatasetValidationReport
    file_count: int


@dataclass(slots=True)
class DatasetBuildResult:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int
    promote_dir: Path | None
    pulled_blocks: bool

    @property
    def reused(self) -> bool:
        return self.promote_dir is None


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


async def build_history_plan(
    *,
    config: ExperimentConfig,
    block_client: Web3BlockClient,
    required_history_blocks: int,
) -> BlockPullPlan:
    return await block_client.plan_history_window(
        end_timestamp=config.history_window_end_timestamp,
        required_history_blocks=required_history_blocks,
        chunk_size=config.acquisition.chunk_size,
    )


def _load_existing_dataset(
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


def _block_range_start(validation: BlockDatasetValidationReport) -> int:
    if validation.first_block_number is None:
        raise ValueError("validated dataset is missing the first block number")
    return validation.first_block_number


def _block_range_end(validation: BlockDatasetValidationReport) -> int:
    if validation.last_block_number is None:
        raise ValueError("validated dataset is missing the last block number")
    return validation.last_block_number + 1


async def _pull_plan_to_frame(
    *,
    block_client: Web3BlockClient,
    plan: BlockPullPlan,
    output_dir: Path,
    chunk_size: int,
    rpc_controller,
    reporter: Reporter,
) -> pl.DataFrame:
    await block_client.pull_block_range(
        output_dir,
        plan=plan,
        chunk_size=chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
    )
    return load_block_frame(output_dir)


def _filter_block_range(frame: pl.DataFrame, block_range: BlockRange) -> pl.DataFrame:
    return frame.filter(
        (pl.col("block_number") >= block_range.start)
        & (pl.col("block_number") < block_range.end)
    ).sort("block_number")


def _write_block_dataset_dir(
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


def _validate_history_result(
    validation: BlockDatasetValidationReport,
    *,
    history_plan: BlockPullPlan,
    required_history_blocks: int,
) -> None:
    if validation.status != "clean":
        raise ValueError(f"Canonical history dataset validation failed: {validation}")
    if validation.row_count < required_history_blocks:
        raise ValueError(
            "History dataset is too short; "
            f"need at least {required_history_blocks} blocks, got {validation.row_count}"
        )
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


def _validate_evaluation_result(
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


def _combined_frame(*frames: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [frame for frame in frames if frame.height > 0],
        how="vertical",
    ).sort("block_number")


def _partial_plan(
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


async def ensure_history_dataset(
    *,
    config: ExperimentConfig,
    block_client: Web3BlockClient,
    output_dir: Path,
    working_dir: Path,
    history_plan: BlockPullPlan,
    required_history_blocks: int,
    rpc_controller,
    reporter: Reporter,
) -> DatasetBuildResult:
    chunk_size = config.acquisition.chunk_size
    existing = _load_existing_dataset(output_dir, expected_chain_id=config.chain.chain_id)
    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean history validation requires an in-memory frame")
        existing_end = _block_range_end(existing.validation)
        if existing_end == history_plan.block_range.end:
            existing_start = _block_range_start(existing.validation)
            if existing_start <= history_plan.block_range.start:
                reporter.log(f"reusing canonical dataset: {output_dir}")
                _validate_history_result(
                    existing.validation,
                    history_plan=history_plan,
                    required_history_blocks=required_history_blocks,
                )
                return DatasetBuildResult(
                    path=output_dir,
                    validation=existing.validation,
                    file_count=existing.file_count,
                    promote_dir=None,
                    pulled_blocks=False,
                )

            reporter.log(f"extending canonical dataset: {output_dir}")
            prefix_plan = _partial_plan(
                block_client,
                start_block=history_plan.block_range.start,
                end_block=existing_start,
                window=history_plan.window,
                chunk_size=chunk_size,
            )
            if prefix_plan is None:
                raise RuntimeError("history prefix plan unexpectedly resolved to empty")
            prefix_frame = await _pull_plan_to_frame(
                block_client=block_client,
                plan=prefix_plan,
                output_dir=working_dir / "history-prefix",
                chunk_size=chunk_size,
                rpc_controller=rpc_controller,
                reporter=reporter,
            )
            history_frame = _combined_frame(prefix_frame, existing.frame)
            file_count = _write_block_dataset_dir(
                working_dir / "history",
                frame=history_frame,
                chunk_size=chunk_size,
                chain_name=config.chain.name.value,
            )
            validation = validate_block_dataset(
                working_dir / "history",
                expected_chain_id=config.chain.chain_id,
            )
            _validate_history_result(
                validation,
                history_plan=history_plan,
                required_history_blocks=required_history_blocks,
            )
            return DatasetBuildResult(
                path=working_dir / "history",
                validation=validation,
                file_count=file_count,
                promote_dir=working_dir / "history",
                pulled_blocks=True,
            )

    if existing is not None:
        reporter.log(f"rebuilding canonical dataset: {output_dir}", level="warning")

    pulled_frame = await _pull_plan_to_frame(
        block_client=block_client,
        plan=history_plan,
        output_dir=working_dir / "history",
        chunk_size=chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
    )
    validation = validate_block_dataset(
        working_dir / "history",
        expected_chain_id=config.chain.chain_id,
    )
    _validate_history_result(
        validation,
        history_plan=history_plan,
        required_history_blocks=required_history_blocks,
    )
    return DatasetBuildResult(
        path=working_dir / "history",
        validation=validation,
        file_count=len(iter_block_files(working_dir / "history")),
        promote_dir=working_dir / "history",
        pulled_blocks=pulled_frame.height > 0,
    )


async def ensure_evaluation_dataset(
    *,
    config: ExperimentConfig,
    block_client: Web3BlockClient,
    output_dir: Path,
    working_dir: Path,
    evaluation_plan: BlockPullPlan,
    rpc_controller,
    reporter: Reporter,
) -> DatasetBuildResult:
    chunk_size = config.acquisition.chunk_size
    existing = _load_existing_dataset(output_dir, expected_chain_id=config.chain.chain_id)
    target_start = evaluation_plan.block_range.start
    target_end = evaluation_plan.block_range.end

    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean evaluation validation requires an in-memory frame")
        existing_start = _block_range_start(existing.validation)
        existing_end = _block_range_end(existing.validation)
        if existing_start == target_start and existing_end == target_end:
            reporter.log(f"reusing canonical dataset: {output_dir}")
            validation = existing.validation.model_copy(deep=True)
            _validate_evaluation_result(
                validation,
                evaluation_dir=output_dir,
                evaluation_plan=evaluation_plan,
                expected_chain_id=config.chain.chain_id,
            )
            return DatasetBuildResult(
                path=output_dir,
                validation=validation,
                file_count=existing.file_count,
                promote_dir=None,
                pulled_blocks=False,
            )

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
            reporter.log(f"reusing overlapping canonical dataset: {output_dir}")
            frames: list[pl.DataFrame] = []
            pulled_blocks = False

            prefix_plan = _partial_plan(
                block_client,
                start_block=target_start,
                end_block=overlap_start,
                window=evaluation_plan.window,
                chunk_size=chunk_size,
            )
            if prefix_plan is not None:
                frames.append(
                    await _pull_plan_to_frame(
                        block_client=block_client,
                        plan=prefix_plan,
                        output_dir=working_dir / "evaluation-prefix",
                        chunk_size=chunk_size,
                        rpc_controller=rpc_controller,
                        reporter=reporter,
                    )
                )
                pulled_blocks = True

            frames.append(
                _filter_block_range(
                    existing.frame,
                    BlockRange(start=overlap_start, end=overlap_end),
                )
            )

            suffix_plan = _partial_plan(
                block_client,
                start_block=overlap_end,
                end_block=target_end,
                window=evaluation_plan.window,
                chunk_size=chunk_size,
            )
            if suffix_plan is not None:
                frames.append(
                    await _pull_plan_to_frame(
                        block_client=block_client,
                        plan=suffix_plan,
                        output_dir=working_dir / "evaluation-suffix",
                        chunk_size=chunk_size,
                        rpc_controller=rpc_controller,
                        reporter=reporter,
                    )
                )
                pulled_blocks = True

            evaluation_frame = _combined_frame(*frames)
            file_count = _write_block_dataset_dir(
                working_dir / "evaluation",
                frame=evaluation_frame,
                chunk_size=chunk_size,
                chain_name=config.chain.name.value,
            )
            validation = validate_block_dataset(
                working_dir / "evaluation",
                expected_chain_id=config.chain.chain_id,
            )
            _validate_evaluation_result(
                validation,
                evaluation_dir=working_dir / "evaluation",
                evaluation_plan=evaluation_plan,
                expected_chain_id=config.chain.chain_id,
            )
            return DatasetBuildResult(
                path=working_dir / "evaluation",
                validation=validation,
                file_count=file_count,
                promote_dir=working_dir / "evaluation",
                pulled_blocks=pulled_blocks,
            )

    if existing is not None:
        reporter.log(f"rebuilding canonical dataset: {output_dir}", level="warning")

    await _pull_plan_to_frame(
        block_client=block_client,
        plan=evaluation_plan,
        output_dir=working_dir / "evaluation",
        chunk_size=chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
    )
    validation = validate_block_dataset(
        working_dir / "evaluation",
        expected_chain_id=config.chain.chain_id,
    )
    _validate_evaluation_result(
        validation,
        evaluation_dir=working_dir / "evaluation",
        evaluation_plan=evaluation_plan,
        expected_chain_id=config.chain.chain_id,
    )
    return DatasetBuildResult(
        path=working_dir / "evaluation",
        validation=validation,
        file_count=len(iter_block_files(working_dir / "evaluation")),
        promote_dir=working_dir / "evaluation",
        pulled_blocks=True,
    )

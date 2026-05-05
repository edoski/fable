"""Corpus split materialization session."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import copy2
from typing import NoReturn, assert_never

import polars as pl

from ...acquisition import (
    AcquisitionPullController,
    BlockPullPlan,
    BlockRange,
    BlockSource,
    TimestampRange,
    pull_block_range,
)
from ...core.files import remove_path
from ..contract import CanonicalBlockRow, canonicalize_block_frame
from ..io import iter_block_files, load_block_frame, write_block_file
from ..metadata import has_block_files
from ..validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)
from ._policy import (
    CorpusSplitOutcome,
    ExtendEvaluationCommittedSplitPlan,
    ExtendHistoryCommittedSplitPlan,
    MaterializeFullSplitPlan,
    RejectInvalidStagedSplitPlan,
    ReuseCommittedSplitPlan,
    ReuseStagedSplitPlan,
    SplitDatasetCandidate,
    SplitDatasetFacts,
    SplitMaterializationPlan,
    SplitPullRange,
    SplitTarget,
    plan_evaluation_split_materialization,
    plan_history_split_materialization,
)


@dataclass(slots=True)
class CorpusSplitMaterializationSpec:
    chain_name: str
    expected_chain_id: int
    chunk_size: int


@dataclass(slots=True)
class ExistingDatasetState:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int


class CorpusSplitKind(StrEnum):
    HISTORY = "history"
    EVALUATION = "evaluation"


@dataclass(slots=True)
class DatasetBuildResult:
    path: Path
    validation: BlockDatasetValidationReport
    file_count: int
    promote_dir: Path | None
    outcome: CorpusSplitOutcome


@dataclass(frozen=True, slots=True)
class CorpusSplitIntent:
    kind: CorpusSplitKind
    output_dir: Path
    working_dir: Path
    plan: BlockPullPlan


StatusCallback = Callable[[str], None]
ValidationCallback = Callable[[BlockDatasetValidationReport, Path], None]


@dataclass(frozen=True, slots=True)
class CorpusSplitMaterializationSession:
    materialization: CorpusSplitMaterializationSpec
    block_source: BlockSource
    controller: AcquisitionPullController
    status: StatusCallback | None = None

    async def fulfill(self, intent: CorpusSplitIntent) -> DatasetBuildResult:
        if intent.kind is CorpusSplitKind.HISTORY:
            return await _ensure_history_split(
                intent,
                materialization=self.materialization,
                block_source=self.block_source,
                controller=self.controller,
                status=self.status,
            )
        if intent.kind is CorpusSplitKind.EVALUATION:
            return await _ensure_evaluation_split(
                intent,
                materialization=self.materialization,
                block_source=self.block_source,
                controller=self.controller,
                status=self.status,
            )
        raise ValueError(f"Unsupported corpus split kind: {intent.kind}")


@dataclass(slots=True)
class _ParquetBlockPullSink:
    output_dir: Path
    materialization: CorpusSplitMaterializationSpec
    pending_rows: list[CanonicalBlockRow]

    @classmethod
    def create(
        cls,
        output_dir: Path,
        *,
        materialization: CorpusSplitMaterializationSpec,
    ) -> _ParquetBlockPullSink:
        return cls(output_dir=output_dir, materialization=materialization, pending_rows=[])

    def completed_prefix_end(self, plan: BlockPullPlan) -> int:
        return completed_prefix_end(
            self.output_dir,
            plan=plan,
            expected_chain_id=self.materialization.expected_chain_id,
        )

    def write_rows(self, rows: list[CanonicalBlockRow]) -> None:
        self.pending_rows.extend(rows)
        while len(self.pending_rows) >= self.materialization.chunk_size:
            write_block_rows_chunk(
                self.output_dir,
                chain_name=self.materialization.chain_name,
                rows=self.pending_rows[: self.materialization.chunk_size],
            )
            self.pending_rows = self.pending_rows[self.materialization.chunk_size :]

    def finish(self) -> None:
        if self.pending_rows:
            write_block_rows_chunk(
                self.output_dir,
                chain_name=self.materialization.chain_name,
                rows=self.pending_rows,
            )
            self.pending_rows = []


def noop_status(message: str) -> None:
    del message


def validate_block_dataset(
    path: Path,
    *,
    expected_chain_id: int,
) -> BlockDatasetValidationReport:
    try:
        frame = load_block_frame(path)
    except Exception as exc:  # pragma: no cover - workflow smoke tests cover this path
        return BlockDatasetValidationReport(dataset_path=path, status="error", errors=[str(exc)])
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
            validation=BlockDatasetValidationReport(
                dataset_path=path,
                status="error",
                errors=[str(exc)],
            ),
            file_count=len(iter_block_files(path)),
        )
    return ExistingDatasetState(
        path=path,
        validation=validate_contiguous_block_frame(
            frame,
            dataset_path=path,
            expected_chain_id=expected_chain_id,
        ),
        file_count=len(iter_block_files(path)),
    )


def reused_result(
    existing: SplitDatasetCandidate,
    *,
    validation: BlockDatasetValidationReport | None = None,
) -> DatasetBuildResult:
    return DatasetBuildResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=None,
        outcome=CorpusSplitOutcome.REUSED,
    )


def staged_result(
    existing: SplitDatasetCandidate,
    *,
    outcome: CorpusSplitOutcome,
    validation: BlockDatasetValidationReport | None = None,
) -> DatasetBuildResult:
    return DatasetBuildResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=existing.path,
        outcome=outcome,
    )


def split_candidate(existing: ExistingDatasetState | None) -> SplitDatasetCandidate | None:
    if existing is None:
        return None
    return SplitDatasetCandidate(
        path=existing.path,
        validation=existing.validation,
        facts=SplitDatasetFacts(
            status=existing.validation.status,
            first_block_number=existing.validation.first_block_number,
            last_block_number=existing.validation.last_block_number,
        ),
        file_count=existing.file_count,
    )


def materialize_dataset(
    *,
    mode: str,
    materialization: CorpusSplitMaterializationSpec,
    working_dir: Path,
    validate_result: ValidationCallback,
    frames: Sequence[pl.DataFrame] | None = None,
    outcome: CorpusSplitOutcome,
) -> DatasetBuildResult:
    dataset_dir = working_dir / mode
    if frames is not None:
        remove_path(dataset_dir)
        file_count = write_block_dataset_dir(
            dataset_dir,
            frame=combined_frame(*frames),
            chunk_size=materialization.chunk_size,
            chain_name=materialization.chain_name,
        )
    else:
        file_count = len(iter_block_files(dataset_dir))
    validation = validate_block_dataset(
        dataset_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    validate_result(validation, dataset_dir)
    return DatasetBuildResult(
        path=dataset_dir,
        validation=validation,
        file_count=file_count,
        promote_dir=dataset_dir,
        outcome=outcome,
    )


def materialize_dataset_from_sources(
    *,
    mode: str,
    materialization: CorpusSplitMaterializationSpec,
    working_dir: Path,
    validate_result: ValidationCallback,
    source_dirs: Sequence[Path] = (),
    source_files: Sequence[Path] = (),
    frames: Sequence[pl.DataFrame] = (),
    outcome: CorpusSplitOutcome,
) -> DatasetBuildResult:
    dataset_dir = working_dir / mode
    remove_path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        for source_file in iter_block_files(source_dir):
            copy2(source_file, dataset_dir / source_file.name)
    for source_file in source_files:
        copy2(source_file, dataset_dir / source_file.name)
    for frame in frames:
        if frame.height > 0:
            write_block_dataset_dir(
                dataset_dir,
                frame=frame,
                chunk_size=materialization.chunk_size,
                chain_name=materialization.chain_name,
            )
    validation = validate_block_dataset(
        dataset_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    validate_result(validation, dataset_dir)
    return DatasetBuildResult(
        path=dataset_dir,
        validation=validation,
        file_count=len(iter_block_files(dataset_dir)),
        promote_dir=dataset_dir,
        outcome=outcome,
    )


async def pull_plan_to_frame(
    *,
    block_source: BlockSource,
    plan: BlockPullPlan,
    output_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> pl.DataFrame:
    await pull_block_range(
        block_source,
        plan=plan,
        controller=controller,
        sink=_ParquetBlockPullSink.create(output_dir, materialization=materialization),
    )
    return load_block_frame(output_dir)


async def pull_plan_to_dir(
    *,
    block_source: BlockSource,
    plan: BlockPullPlan,
    output_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> Path:
    await pull_block_range(
        block_source,
        plan=plan,
        controller=controller,
        sink=_ParquetBlockPullSink.create(output_dir, materialization=materialization),
    )
    return output_dir


def plan_pull_dir(working_dir: Path, *, label: str, plan: BlockPullPlan) -> Path:
    return (
        working_dir
        / "pulls"
        / f"{label}__{plan.block_range.start}_to_{plan.block_range.end}"
    )


def filter_block_range(frame: pl.DataFrame, block_range: BlockRange) -> pl.DataFrame:
    return frame.filter(
        (pl.col("block_number") >= block_range.start)
        & (pl.col("block_number") < block_range.end)
    ).sort("block_number")


def reusable_block_files_and_edges(
    dataset_dir: Path,
    *,
    block_range: BlockRange,
) -> tuple[list[Path], list[pl.DataFrame]]:
    reusable_files: list[Path] = []
    edge_frames: list[pl.DataFrame] = []
    for block_file in iter_block_files(dataset_dir):
        frame = load_block_frame(block_file)
        start_block = int(frame["block_number"][0])
        end_block = int(frame["block_number"][-1]) + 1
        if end_block <= block_range.start or start_block >= block_range.end:
            continue
        if start_block >= block_range.start and end_block <= block_range.end:
            reusable_files.append(block_file)
            continue
        edge = filter_block_range(frame, block_range)
        if edge.height > 0:
            edge_frames.append(edge)
    return reusable_files, edge_frames


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


def completed_prefix_end(
    output_dir: Path,
    *,
    plan: BlockPullPlan,
    expected_chain_id: int,
) -> int:
    if not output_dir.exists():
        return plan.block_range.start
    try:
        frame = load_block_frame(output_dir)
    except ValueError as exc:
        if "No parquet block files found" in str(exc):
            return plan.block_range.start
        raise RuntimeError(
            f"Cannot resume from invalid partial block dataset: {output_dir}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume from invalid partial block dataset: {output_dir}"
        ) from exc

    validation = validate_contiguous_block_frame(
        frame,
        dataset_path=output_dir,
        expected_chain_id=expected_chain_id,
    )
    if validation.status != "clean":
        raise RuntimeError(f"Cannot resume from invalid partial block dataset: {validation}")
    if validation.first_block_number != plan.block_range.start:
        raise RuntimeError(
            "Cannot resume partial block dataset with a different start block: "
            f"expected {plan.block_range.start}, got {validation.first_block_number}"
        )
    if validation.last_block_number is None:
        return plan.block_range.start
    if validation.last_block_number >= plan.block_range.end:
        return plan.block_range.end
    return validation.last_block_number + 1


def write_block_rows_chunk(
    output_dir: Path,
    *,
    chain_name: str,
    rows: Sequence[CanonicalBlockRow],
) -> Path:
    frame = canonicalize_block_frame(pl.DataFrame(rows))
    start_block = int(frame["block_number"][0])
    end_block = int(frame["block_number"][-1])
    destination = output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet"
    write_block_file(destination, frame)
    return destination


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


def combined_frame(*frames: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([frame for frame in frames if frame.height > 0], how="vertical").sort(
        "block_number"
    )


def pull_range_plan(
    block_source: BlockSource,
    pull_range: SplitPullRange,
    *,
    window: TimestampRange,
) -> BlockPullPlan:
    return block_source.plan_block_range(pull_range.block_range, window=window)


async def pull_plan_range_to_dir(
    *,
    block_source: BlockSource,
    pull_range: SplitPullRange,
    window: TimestampRange,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    controller: AcquisitionPullController,
) -> Path:
    plan = pull_range_plan(block_source, pull_range, window=window)
    return await pull_plan_to_dir(
        block_source=block_source,
        plan=plan,
        output_dir=plan_pull_dir(working_dir, label=pull_range.label, plan=plan),
        materialization=materialization,
        controller=controller,
    )


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


def reject_invalid_staged(plan: RejectInvalidStagedSplitPlan) -> NoReturn:
    raise RuntimeError(f"{plan.error_message}: {plan.dataset.validation}")


@dataclass(frozen=True, slots=True)
class _SplitPlanExecution:
    kind: CorpusSplitKind
    plan: BlockPullPlan
    working_dir: Path
    materialization: CorpusSplitMaterializationSpec
    block_source: BlockSource
    controller: AcquisitionPullController
    emit: StatusCallback
    validate_result: ValidationCallback


async def _execute_split_materialization_plan(
    plan: SplitMaterializationPlan,
    *,
    execution: _SplitPlanExecution,
) -> DatasetBuildResult:
    if isinstance(plan, RejectInvalidStagedSplitPlan):
        reject_invalid_staged(plan)

    if isinstance(plan, ReuseStagedSplitPlan):
        execution.emit(plan.status_message)
        return staged_result(
            plan.dataset,
            validation=plan.validation,
            outcome=plan.outcome,
        )

    if isinstance(plan, ReuseCommittedSplitPlan):
        execution.emit(plan.status_message)
        return reused_result(plan.dataset, validation=plan.validation)

    if isinstance(plan, ExtendHistoryCommittedSplitPlan):
        return await _extend_history_committed_split(plan, execution=execution)

    if isinstance(plan, ExtendEvaluationCommittedSplitPlan):
        return await _extend_evaluation_committed_split(plan, execution=execution)

    if isinstance(plan, MaterializeFullSplitPlan):
        return await _materialize_full_split(plan, execution=execution)

    assert_never(plan)


async def _extend_history_committed_split(
    plan: ExtendHistoryCommittedSplitPlan,
    *,
    execution: _SplitPlanExecution,
) -> DatasetBuildResult:
    execution.emit(plan.status_message)
    prefix_dir = await pull_plan_range_to_dir(
        block_source=execution.block_source,
        pull_range=plan.prefix,
        window=execution.plan.window,
        working_dir=execution.working_dir,
        materialization=execution.materialization,
        controller=execution.controller,
    )
    return materialize_dataset_from_sources(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        source_dirs=(prefix_dir, plan.existing.path),
        outcome=CorpusSplitOutcome.EXTENDED,
    )


async def _extend_evaluation_committed_split(
    plan: ExtendEvaluationCommittedSplitPlan,
    *,
    execution: _SplitPlanExecution,
) -> DatasetBuildResult:
    source_dirs: list[Path] = []
    source_files, frames = reusable_block_files_and_edges(
        plan.existing.path,
        block_range=plan.reusable_range,
    )
    for pull_range in plan.pull_ranges:
        source_dirs.append(
            await pull_plan_range_to_dir(
                block_source=execution.block_source,
                pull_range=pull_range,
                window=execution.plan.window,
                working_dir=execution.working_dir,
                materialization=execution.materialization,
                controller=execution.controller,
            )
        )
    execution.emit(plan.status_message)
    return materialize_dataset_from_sources(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        source_dirs=source_dirs,
        source_files=source_files,
        frames=frames,
        outcome=CorpusSplitOutcome.EXTENDED,
    )


async def _materialize_full_split(
    plan: MaterializeFullSplitPlan,
    *,
    execution: _SplitPlanExecution,
) -> DatasetBuildResult:
    execution.emit(plan.status_message)
    frame = await pull_plan_to_frame(
        block_source=execution.block_source,
        plan=execution.plan,
        output_dir=plan_pull_dir(
            execution.working_dir,
            label=execution.kind.value,
            plan=execution.plan,
        ),
        materialization=execution.materialization,
        controller=execution.controller,
    )
    return materialize_dataset(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        frames=(frame,),
        outcome=plan.outcome,
    )


async def _ensure_history_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> DatasetBuildResult:
    history_plan = intent.plan
    output_dir = intent.output_dir
    working_dir = intent.working_dir
    emit = status or noop_status
    existing = load_existing_dataset(
        output_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    staged = load_existing_dataset(
        working_dir / "history",
        expected_chain_id=materialization.expected_chain_id,
    )

    def validate_result(validation: BlockDatasetValidationReport, _: Path) -> None:
        validate_history_result(validation, history_plan=history_plan)

    materialization_plan = plan_history_split_materialization(
        SplitTarget(
            kind=intent.kind.value,
            block_range=history_plan.block_range,
            window=history_plan.window,
        ),
        existing=split_candidate(existing),
        staged=split_candidate(staged),
        validate_target=validate_result,
    )

    return await _execute_split_materialization_plan(
        materialization_plan,
        execution=_SplitPlanExecution(
            kind=intent.kind,
            plan=history_plan,
            working_dir=working_dir,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            emit=emit,
            validate_result=validate_result,
        ),
    )


async def _ensure_evaluation_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> DatasetBuildResult:
    evaluation_plan = intent.plan
    output_dir = intent.output_dir
    working_dir = intent.working_dir
    emit = status or noop_status
    existing = load_existing_dataset(
        output_dir,
        expected_chain_id=materialization.expected_chain_id,
    )
    staged = load_existing_dataset(
        working_dir / "evaluation",
        expected_chain_id=materialization.expected_chain_id,
    )

    def validate_result(validation: BlockDatasetValidationReport, dataset_dir: Path) -> None:
        validate_evaluation_result(
            validation,
            evaluation_dir=dataset_dir,
            evaluation_plan=evaluation_plan,
            expected_chain_id=materialization.expected_chain_id,
        )

    materialization_plan = plan_evaluation_split_materialization(
        SplitTarget(
            kind=intent.kind.value,
            block_range=evaluation_plan.block_range,
            window=evaluation_plan.window,
        ),
        existing=split_candidate(existing),
        staged=split_candidate(staged),
        validate_target=validate_result,
        reusable_range_matches_target_window=reusable_range_matches_target_window,
    )

    return await _execute_split_materialization_plan(
        materialization_plan,
        execution=_SplitPlanExecution(
            kind=intent.kind,
            plan=evaluation_plan,
            working_dir=working_dir,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            emit=emit,
            validate_result=validate_result,
        ),
    )

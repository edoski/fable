"""Corpus split materialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import copy2

import polars as pl

from ..acquisition import (
    AcquisitionPullController,
    BlockPullPlan,
    BlockRange,
    BlockSource,
    TimestampRange,
    pull_block_range,
)
from ..core.files import remove_path
from .contract import CanonicalBlockRow, canonicalize_block_frame
from .io import iter_block_files, load_block_frame, write_block_file
from .metadata import has_block_files
from .validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
    validate_exact_window_frame,
)


@dataclass(slots=True)
class CorpusSplitMaterializationSpec:
    chain_name: str
    expected_chain_id: int
    chunk_size: int


@dataclass(slots=True)
class ExistingDatasetState:
    path: Path
    frame: pl.DataFrame | None
    validation: BlockDatasetValidationReport
    file_count: int


class CorpusSplitOutcome(StrEnum):
    CREATED = "created"
    REUSED = "reused"
    EXTENDED = "extended"
    REBUILT = "rebuilt"


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
            frame=None,
            validation=BlockDatasetValidationReport(
                dataset_path=path,
                status="error",
                errors=[str(exc)],
            ),
            file_count=len(iter_block_files(path)),
        )
    return ExistingDatasetState(
        path=path,
        frame=frame,
        validation=validate_contiguous_block_frame(
            frame,
            dataset_path=path,
            expected_chain_id=expected_chain_id,
        ),
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


def reused_result(
    existing: ExistingDatasetState,
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
    existing: ExistingDatasetState,
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


def partial_plan(
    block_source: BlockSource,
    *,
    start_block: int,
    end_block: int,
    window: TimestampRange,
) -> BlockPullPlan | None:
    if end_block <= start_block:
        return None
    return block_source.plan_block_range(
        BlockRange(start=start_block, end=end_block),
        window=window,
    )


def history_split_intent(
    *,
    output_dir: Path,
    working_dir: Path,
    history_plan: BlockPullPlan,
) -> CorpusSplitIntent:
    return CorpusSplitIntent(
        kind=CorpusSplitKind.HISTORY,
        output_dir=output_dir,
        working_dir=working_dir,
        plan=history_plan,
    )


def evaluation_split_intent(
    *,
    output_dir: Path,
    working_dir: Path,
    evaluation_plan: BlockPullPlan,
) -> CorpusSplitIntent:
    return CorpusSplitIntent(
        kind=CorpusSplitKind.EVALUATION,
        output_dir=output_dir,
        working_dir=working_dir,
        plan=evaluation_plan,
    )


async def ensure_corpus_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None = None,
) -> DatasetBuildResult:
    if intent.kind is CorpusSplitKind.HISTORY:
        return await _ensure_history_split(
            intent,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            status=status,
        )
    if intent.kind is CorpusSplitKind.EVALUATION:
        return await _ensure_evaluation_split(
            intent,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            status=status,
        )
    raise ValueError(f"Unsupported corpus split kind: {intent.kind}")


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

    if staged is not None:
        if staged.validation.status != "clean":
            raise RuntimeError(f"Cannot resume invalid staged history dataset: {staged.validation}")
        staged_validation = staged.validation.model_copy(deep=True)
        try:
            validate_result(staged_validation, staged.path)
        except ValueError:
            pass
        else:
            emit("history reused staged dataset")
            return staged_result(
                staged,
                validation=staged_validation,
                outcome=(
                    CorpusSplitOutcome.REBUILT
                    if existing is not None
                    else CorpusSplitOutcome.CREATED
                ),
            )

    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean history validation requires an in-memory frame")
        existing_end = block_range_end(existing.validation)
        if existing_end == history_plan.block_range.end:
            existing_start = block_range_start(existing.validation)
            if existing_start <= history_plan.block_range.start:
                emit("history reused cached dataset")
                validate_history_result(existing.validation, history_plan=history_plan)
                return reused_result(existing)

            prefix_plan = partial_plan(
                block_source,
                start_block=history_plan.block_range.start,
                end_block=existing_start,
                window=history_plan.window,
            )
            if prefix_plan is None:
                raise RuntimeError("history prefix plan unexpectedly resolved to empty")
            emit("history extending cached dataset")
            prefix_dir = await pull_plan_to_dir(
                block_source=block_source,
                plan=prefix_plan,
                output_dir=plan_pull_dir(
                    working_dir,
                    label="history-prefix",
                    plan=prefix_plan,
                ),
                materialization=materialization,
                controller=controller,
            )
            return materialize_dataset_from_sources(
                mode="history",
                materialization=materialization,
                working_dir=working_dir,
                validate_result=validate_result,
                source_dirs=(prefix_dir, existing.path),
                outcome=CorpusSplitOutcome.EXTENDED,
            )

    emit("history downloading")
    frame = await pull_plan_to_frame(
        block_source=block_source,
        plan=history_plan,
        output_dir=plan_pull_dir(working_dir, label="history", plan=history_plan),
        materialization=materialization,
        controller=controller,
    )
    return materialize_dataset(
        mode="history",
        materialization=materialization,
        working_dir=working_dir,
        validate_result=validate_result,
        frames=(frame,),
        outcome=(
            CorpusSplitOutcome.REBUILT
            if existing is not None
            else CorpusSplitOutcome.CREATED
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

    if staged is not None:
        if staged.validation.status != "clean":
            raise RuntimeError(
                f"Cannot resume invalid staged evaluation dataset: {staged.validation}"
            )
        staged_validation = staged.validation.model_copy(deep=True)
        try:
            validate_result(staged_validation, staged.path)
        except ValueError:
            pass
        else:
            emit("evaluation reused staged dataset")
            return staged_result(
                staged,
                validation=staged_validation,
                outcome=(
                    CorpusSplitOutcome.REBUILT
                    if existing is not None
                    else CorpusSplitOutcome.CREATED
                ),
            )

    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean evaluation validation requires an in-memory frame")
        existing_start = block_range_start(existing.validation)
        existing_end = block_range_end(existing.validation)
        target_start = evaluation_plan.block_range.start
        target_end = evaluation_plan.block_range.end

        if existing_start == target_start and existing_end == target_end:
            validation = existing.validation.model_copy(deep=True)
            emit("evaluation reused cached dataset")
            validate_result(validation, existing.path)
            return reused_result(existing, validation=validation)

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
            source_dirs: list[Path] = []
            source_files, frames = reusable_block_files_and_edges(
                existing.path,
                block_range=BlockRange(start=overlap_start, end=overlap_end),
            )

            prefix_plan = partial_plan(
                block_source,
                start_block=target_start,
                end_block=overlap_start,
                window=evaluation_plan.window,
            )
            if prefix_plan is not None:
                source_dirs.append(
                    await pull_plan_to_dir(
                        block_source=block_source,
                        plan=prefix_plan,
                        output_dir=plan_pull_dir(
                            working_dir,
                            label="evaluation-prefix",
                            plan=prefix_plan,
                        ),
                        materialization=materialization,
                        controller=controller,
                    )
                )

            suffix_plan = partial_plan(
                block_source,
                start_block=overlap_end,
                end_block=target_end,
                window=evaluation_plan.window,
            )
            if suffix_plan is not None:
                source_dirs.append(
                    await pull_plan_to_dir(
                        block_source=block_source,
                        plan=suffix_plan,
                        output_dir=plan_pull_dir(
                            working_dir,
                            label="evaluation-suffix",
                            plan=suffix_plan,
                        ),
                        materialization=materialization,
                        controller=controller,
                    )
                )

            emit("evaluation extending cached dataset")
            return materialize_dataset_from_sources(
                mode="evaluation",
                materialization=materialization,
                working_dir=working_dir,
                validate_result=validate_result,
                source_dirs=source_dirs,
                source_files=source_files,
                frames=frames,
                outcome=CorpusSplitOutcome.EXTENDED,
            )

    emit("evaluation downloading")
    frame = await pull_plan_to_frame(
        block_source=block_source,
        plan=evaluation_plan,
        output_dir=plan_pull_dir(working_dir, label="evaluation", plan=evaluation_plan),
        materialization=materialization,
        controller=controller,
    )
    return materialize_dataset(
        mode="evaluation",
        materialization=materialization,
        working_dir=working_dir,
        validate_result=validate_result,
        frames=(frame,),
        outcome=(
            CorpusSplitOutcome.REBUILT
            if existing is not None
            else CorpusSplitOutcome.CREATED
        ),
    )

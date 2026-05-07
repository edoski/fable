"""Corpus split materialization session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...acquisition import AcquisitionPullController, BlockPullPlan, BlockRange, BlockSource
from ..validation import BlockDatasetValidationReport
from ._dataset_state import load_split_candidate, reused_result, staged_result
from ._invariants import (
    reusable_range_matches_target_window,
    validate_evaluation_result,
    validate_history_result,
)
from ._models import (
    CorpusSplitIntent,
    CorpusSplitKind,
    CorpusSplitMaterializationResult,
    CorpusSplitMaterializationSpec,
    CorpusSplitOutcome,
    StatusCallback,
    ValidationCallback,
    _SplitDatasetCandidate,
    _SplitDatasetFacts,
    _SplitPullRange,
)
from ._pulls import plan_pull_dir, pull_plan_range_to_dir, pull_plan_to_frame
from ._reuse import (
    materialize_dataset,
    materialize_dataset_from_sources,
    reusable_block_files_and_edges,
)


@dataclass(frozen=True, slots=True)
class CorpusSplitMaterializationSession:
    materialization: CorpusSplitMaterializationSpec
    block_source: BlockSource
    controller: AcquisitionPullController
    status: StatusCallback | None = None

    async def fulfill(
        self,
        intent: CorpusSplitIntent,
    ) -> CorpusSplitMaterializationResult:
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


def noop_status(message: str) -> None:
    del message


async def _ensure_history_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> CorpusSplitMaterializationResult:
    history_plan = intent.plan
    emit = status or noop_status
    existing = load_split_candidate(
        intent.output_dir,
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )
    staged = load_split_candidate(
        intent.working_dir / "history",
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )

    def validate_result(validation: BlockDatasetValidationReport, _: Path) -> None:
        validate_history_result(validation, history_plan=history_plan)

    return await _materialize_history_split(
        existing=existing,
        staged=staged,
        plan=history_plan,
        kind=intent.kind,
        working_dir=intent.working_dir,
        materialization=materialization,
        block_source=block_source,
        controller=controller,
        emit=emit,
        validate_result=validate_result,
    )


async def _ensure_evaluation_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> CorpusSplitMaterializationResult:
    evaluation_plan = intent.plan
    emit = status or noop_status
    existing = load_split_candidate(
        intent.output_dir,
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )
    staged = load_split_candidate(
        intent.working_dir / "evaluation",
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )

    def validate_result(validation: BlockDatasetValidationReport, dataset_dir: Path) -> None:
        validate_evaluation_result(
            validation,
            evaluation_dir=dataset_dir,
            evaluation_plan=evaluation_plan,
            expected_chain_id=materialization.expected_chain_id,
            required_columns=materialization.required_columns,
        )

    return await _materialize_evaluation_split(
        existing=existing,
        staged=staged,
        plan=evaluation_plan,
        kind=intent.kind,
        working_dir=intent.working_dir,
        materialization=materialization,
        block_source=block_source,
        controller=controller,
        emit=emit,
        validate_result=validate_result,
    )


async def _materialize_history_split(
    *,
    existing: _SplitDatasetCandidate | None,
    staged: _SplitDatasetCandidate | None,
    plan: BlockPullPlan,
    kind: CorpusSplitKind,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    emit: StatusCallback,
    validate_result: ValidationCallback,
) -> CorpusSplitMaterializationResult:
    staged_reuse = _staged_reuse_result(
        "history",
        staged=staged,
        existing=existing,
        validate_result=validate_result,
    )
    if staged_reuse is not None:
        emit("history reused staged dataset")
        return staged_reuse

    if existing is not None and existing.facts.status == "clean":
        existing_start = _required_first_block(existing.facts)
        existing_end = _required_end_block(existing.facts)
        target_start = plan.block_range.start
        target_end = plan.block_range.end

        if existing_end == target_end and existing_start <= target_start:
            existing_validation = _target_validation(existing, validate_result)
            if existing_validation is not None:
                emit("history reused committed dataset")
                return reused_result(existing, validation=existing_validation)

        if existing_end == target_end and existing_start > target_start:
            emit("history extending committed dataset")
            prefix_dir = await pull_plan_range_to_dir(
                block_source=block_source,
                pull_range=_SplitPullRange(
                    label="history-prefix",
                    block_range=BlockRange(start=target_start, end=existing_start),
                ),
                window=plan.window,
                working_dir=working_dir,
                materialization=materialization,
                controller=controller,
            )
            return materialize_dataset_from_sources(
                mode=kind.value,
                materialization=materialization,
                working_dir=working_dir,
                validate_result=validate_result,
                source_dirs=(prefix_dir, existing.path),
                outcome=CorpusSplitOutcome.EXTENDED,
            )

    return await _materialize_full_split(
        "history",
        existing=existing,
        kind=kind,
        plan=plan,
        working_dir=working_dir,
        materialization=materialization,
        block_source=block_source,
        controller=controller,
        emit=emit,
        validate_result=validate_result,
    )


async def _materialize_evaluation_split(
    *,
    existing: _SplitDatasetCandidate | None,
    staged: _SplitDatasetCandidate | None,
    plan: BlockPullPlan,
    kind: CorpusSplitKind,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    emit: StatusCallback,
    validate_result: ValidationCallback,
) -> CorpusSplitMaterializationResult:
    staged_reuse = _staged_reuse_result(
        "evaluation",
        staged=staged,
        existing=existing,
        validate_result=validate_result,
    )
    if staged_reuse is not None:
        emit("evaluation reused staged dataset")
        return staged_reuse

    if existing is not None and existing.facts.status == "clean":
        existing_start = _required_first_block(existing.facts)
        existing_end = _required_end_block(existing.facts)
        target_start = plan.block_range.start
        target_end = plan.block_range.end

        if existing_start == target_start and existing_end == target_end:
            existing_validation = _target_validation(existing, validate_result)
            if existing_validation is not None:
                emit("evaluation reused committed dataset")
                return reused_result(existing, validation=existing_validation)

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
            reusable_range = BlockRange(start=overlap_start, end=overlap_end)
            if reusable_range_matches_target_window(
                existing,
                reusable_range,
                plan.window,
            ):
                pull_ranges: list[_SplitPullRange] = []
                if target_start < overlap_start:
                    pull_ranges.append(
                        _SplitPullRange(
                            label="evaluation-prefix",
                            block_range=BlockRange(start=target_start, end=overlap_start),
                        )
                    )
                if overlap_end < target_end:
                    pull_ranges.append(
                        _SplitPullRange(
                            label="evaluation-suffix",
                            block_range=BlockRange(start=overlap_end, end=target_end),
                        )
                    )
                return await _extend_evaluation_committed_split(
                    existing=existing,
                    reusable_range=reusable_range,
                    pull_ranges=tuple(pull_ranges),
                    kind=kind,
                    plan=plan,
                    working_dir=working_dir,
                    materialization=materialization,
                    block_source=block_source,
                    controller=controller,
                    emit=emit,
                    validate_result=validate_result,
                )

    return await _materialize_full_split(
        "evaluation",
        existing=existing,
        kind=kind,
        plan=plan,
        working_dir=working_dir,
        materialization=materialization,
        block_source=block_source,
        controller=controller,
        emit=emit,
        validate_result=validate_result,
    )


async def _extend_evaluation_committed_split(
    *,
    existing: _SplitDatasetCandidate,
    reusable_range: BlockRange,
    pull_ranges: tuple[_SplitPullRange, ...],
    kind: CorpusSplitKind,
    plan: BlockPullPlan,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    emit: StatusCallback,
    validate_result: ValidationCallback,
) -> CorpusSplitMaterializationResult:
    source_dirs: list[Path] = []
    source_files, frames = reusable_block_files_and_edges(
        existing.path,
        block_range=reusable_range,
    )
    for pull_range in pull_ranges:
        source_dirs.append(
            await pull_plan_range_to_dir(
                block_source=block_source,
                pull_range=pull_range,
                window=plan.window,
                working_dir=working_dir,
                materialization=materialization,
                controller=controller,
            )
        )
    emit("evaluation extending committed dataset")
    return materialize_dataset_from_sources(
        mode=kind.value,
        materialization=materialization,
        working_dir=working_dir,
        validate_result=validate_result,
        source_dirs=source_dirs,
        source_files=source_files,
        frames=frames,
        outcome=CorpusSplitOutcome.EXTENDED,
    )


async def _materialize_full_split(
    label: str,
    *,
    existing: _SplitDatasetCandidate | None,
    kind: CorpusSplitKind,
    plan: BlockPullPlan,
    working_dir: Path,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    emit: StatusCallback,
    validate_result: ValidationCallback,
) -> CorpusSplitMaterializationResult:
    emit(f"{label} downloading")
    frame = await pull_plan_to_frame(
        block_source=block_source,
        plan=plan,
        output_dir=plan_pull_dir(working_dir, label=kind.value, plan=plan),
        materialization=materialization,
        controller=controller,
    )
    return materialize_dataset(
        mode=kind.value,
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


def _staged_reuse_result(
    kind: str,
    *,
    staged: _SplitDatasetCandidate | None,
    existing: _SplitDatasetCandidate | None,
    validate_result: ValidationCallback,
) -> CorpusSplitMaterializationResult | None:
    if staged is None:
        return None
    if staged.facts.status != "clean":
        raise RuntimeError(f"Cannot resume invalid staged {kind} dataset: {staged.validation}")
    staged_validation = _target_validation(staged, validate_result)
    if staged_validation is None:
        return None
    return staged_result(
        staged,
        validation=staged_validation,
        outcome=(
            CorpusSplitOutcome.REBUILT
            if existing is not None
            else CorpusSplitOutcome.CREATED
        ),
    )


def _target_validation(
    candidate: _SplitDatasetCandidate,
    validate_target: Callable[[BlockDatasetValidationReport, Path], None],
) -> BlockDatasetValidationReport | None:
    if candidate.facts.status != "clean":
        return None
    validation = candidate.validation.model_copy(deep=True)
    try:
        validate_target(validation, candidate.path)
    except ValueError:
        return None
    return validation


def _required_first_block(facts: _SplitDatasetFacts) -> int:
    if facts.first_block_number is None:
        raise ValueError("validated dataset is missing the first block number")
    return facts.first_block_number


def _required_end_block(facts: _SplitDatasetFacts) -> int:
    if facts.last_block_number is None:
        raise ValueError("validated dataset is missing the last block number")
    return facts.last_block_number + 1

"""Internal corpus split materialization policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from ...acquisition import BlockRange, TimestampRange
from ..validation import BlockDatasetValidationReport


class CorpusSplitOutcome(StrEnum):
    CREATED = "created"
    REUSED = "reused"
    EXTENDED = "extended"
    REBUILT = "rebuilt"


@dataclass(frozen=True, slots=True)
class SplitTarget:
    kind: str
    block_range: BlockRange
    window: TimestampRange


@dataclass(frozen=True, slots=True)
class SplitDatasetFacts:
    status: str
    first_block_number: int | None
    last_block_number: int | None


@dataclass(frozen=True, slots=True)
class SplitDatasetCandidate:
    path: Path
    validation: BlockDatasetValidationReport
    facts: SplitDatasetFacts
    file_count: int


@dataclass(frozen=True, slots=True)
class SplitPullRange:
    label: str
    block_range: BlockRange


TargetValidator = Callable[[BlockDatasetValidationReport, Path], None]
ReusableRangeValidator = Callable[[SplitDatasetCandidate, BlockRange, TimestampRange], bool]


@dataclass(frozen=True, slots=True)
class ReuseStagedSplitPlan:
    dataset: SplitDatasetCandidate
    validation: BlockDatasetValidationReport
    outcome: CorpusSplitOutcome
    status_message: str


@dataclass(frozen=True, slots=True)
class ReuseCommittedSplitPlan:
    dataset: SplitDatasetCandidate
    validation: BlockDatasetValidationReport
    status_message: str


@dataclass(frozen=True, slots=True)
class ExtendHistoryCommittedSplitPlan:
    existing: SplitDatasetCandidate
    prefix: SplitPullRange
    status_message: str


@dataclass(frozen=True, slots=True)
class ExtendEvaluationCommittedSplitPlan:
    existing: SplitDatasetCandidate
    reusable_range: BlockRange
    pull_ranges: tuple[SplitPullRange, ...]
    status_message: str


@dataclass(frozen=True, slots=True)
class MaterializeFullSplitPlan:
    outcome: CorpusSplitOutcome
    status_message: str


@dataclass(frozen=True, slots=True)
class RejectInvalidStagedSplitPlan:
    dataset: SplitDatasetCandidate
    error_message: str


SplitMaterializationPlan: TypeAlias = (
    ReuseStagedSplitPlan
    | ReuseCommittedSplitPlan
    | ExtendHistoryCommittedSplitPlan
    | ExtendEvaluationCommittedSplitPlan
    | MaterializeFullSplitPlan
    | RejectInvalidStagedSplitPlan
)


def plan_history_split_materialization(
    target: SplitTarget,
    *,
    existing: SplitDatasetCandidate | None,
    staged: SplitDatasetCandidate | None,
    validate_target: TargetValidator,
) -> SplitMaterializationPlan:
    if staged is not None:
        invalid = _invalid_staged_plan("history", staged)
        if invalid is not None:
            return invalid
        staged_validation = _target_validation(staged, validate_target)
        if staged_validation is not None:
            return ReuseStagedSplitPlan(
                dataset=staged,
                outcome=_staged_outcome(existing),
                status_message="history reused staged dataset",
                validation=staged_validation,
            )

    if existing is not None and existing.facts.status == "clean":
        existing_start = _required_first_block(existing.facts)
        existing_end = _required_end_block(existing.facts)
        target_start = target.block_range.start
        target_end = target.block_range.end

        if existing_end == target_end and existing_start <= target_start:
            existing_validation = _target_validation(existing, validate_target)
            if existing_validation is None:
                return _full_materialization_plan("history", existing=existing)
            return ReuseCommittedSplitPlan(
                dataset=existing,
                status_message="history reused committed dataset",
                validation=existing_validation,
            )

        if existing_end == target_end and existing_start > target_start:
            return ExtendHistoryCommittedSplitPlan(
                existing=existing,
                status_message="history extending committed dataset",
                prefix=SplitPullRange(
                    label="history-prefix",
                    block_range=BlockRange(start=target_start, end=existing_start),
                ),
            )

    return _full_materialization_plan("history", existing=existing)


def plan_evaluation_split_materialization(
    target: SplitTarget,
    *,
    existing: SplitDatasetCandidate | None,
    staged: SplitDatasetCandidate | None,
    validate_target: TargetValidator,
    reusable_range_matches_target_window: ReusableRangeValidator,
) -> SplitMaterializationPlan:
    if staged is not None:
        invalid = _invalid_staged_plan("evaluation", staged)
        if invalid is not None:
            return invalid
        staged_validation = _target_validation(staged, validate_target)
        if staged_validation is not None:
            return ReuseStagedSplitPlan(
                dataset=staged,
                outcome=_staged_outcome(existing),
                status_message="evaluation reused staged dataset",
                validation=staged_validation,
            )

    if existing is not None and existing.facts.status == "clean":
        existing_start = _required_first_block(existing.facts)
        existing_end = _required_end_block(existing.facts)
        target_start = target.block_range.start
        target_end = target.block_range.end

        if existing_start == target_start and existing_end == target_end:
            existing_validation = _target_validation(existing, validate_target)
            if existing_validation is None:
                return _full_materialization_plan("evaluation", existing=existing)
            return ReuseCommittedSplitPlan(
                dataset=existing,
                status_message="evaluation reused committed dataset",
                validation=existing_validation,
            )

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
            reusable_range = BlockRange(start=overlap_start, end=overlap_end)
            if not reusable_range_matches_target_window(
                existing,
                reusable_range,
                target.window,
            ):
                return _full_materialization_plan("evaluation", existing=existing)
            pull_ranges: list[SplitPullRange] = []
            if target_start < overlap_start:
                pull_ranges.append(
                    SplitPullRange(
                        label="evaluation-prefix",
                        block_range=BlockRange(start=target_start, end=overlap_start),
                    )
                )
            if overlap_end < target_end:
                pull_ranges.append(
                    SplitPullRange(
                        label="evaluation-suffix",
                        block_range=BlockRange(start=overlap_end, end=target_end),
                    )
                )
            return ExtendEvaluationCommittedSplitPlan(
                existing=existing,
                status_message="evaluation extending committed dataset",
                pull_ranges=tuple(pull_ranges),
                reusable_range=reusable_range,
            )

    return _full_materialization_plan("evaluation", existing=existing)


def _invalid_staged_plan(
    kind: str,
    staged: SplitDatasetCandidate,
) -> RejectInvalidStagedSplitPlan | None:
    if staged.facts.status == "clean":
        return None
    return RejectInvalidStagedSplitPlan(
        dataset=staged,
        error_message=f"Cannot resume invalid staged {kind} dataset",
    )


def _full_materialization_plan(
    kind: str,
    *,
    existing: SplitDatasetCandidate | None,
) -> MaterializeFullSplitPlan:
    return MaterializeFullSplitPlan(
        outcome=(
            CorpusSplitOutcome.REBUILT
            if existing is not None
            else CorpusSplitOutcome.CREATED
        ),
        status_message=f"{kind} downloading",
    )


def _staged_outcome(existing: SplitDatasetCandidate | None) -> CorpusSplitOutcome:
    return (
        CorpusSplitOutcome.REBUILT
        if existing is not None
        else CorpusSplitOutcome.CREATED
    )


def _target_validation(
    candidate: SplitDatasetCandidate,
    validate_target: TargetValidator,
) -> BlockDatasetValidationReport | None:
    if candidate.facts.status != "clean":
        return None
    validation = candidate.validation.model_copy(deep=True)
    try:
        validate_target(validation, candidate.path)
    except ValueError:
        return None
    return validation


def _required_first_block(facts: SplitDatasetFacts) -> int:
    if facts.first_block_number is None:
        raise ValueError("validated dataset is missing the first block number")
    return facts.first_block_number


def _required_end_block(facts: SplitDatasetFacts) -> int:
    if facts.last_block_number is None:
        raise ValueError("validated dataset is missing the last block number")
    return facts.last_block_number + 1

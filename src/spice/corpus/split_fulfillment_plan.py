"""Pure corpus split fulfillment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..acquisition import BlockRange, TimestampRange
from .validation import BlockDatasetValidationReport


class SplitFulfillmentAction(StrEnum):
    REUSE_STAGED = "reuse_staged"
    REUSE_CACHED = "reuse_cached"
    EXTEND_CACHED = "extend_cached"
    MATERIALIZE_FULL = "materialize_full"
    REJECT_INVALID_STAGED = "reject_invalid_staged"


class SplitFulfillmentOutcome(StrEnum):
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
    path: Path
    status: str
    first_block_number: int | None
    last_block_number: int | None


@dataclass(frozen=True, slots=True)
class SplitPullRange:
    label: str
    block_range: BlockRange


@dataclass(frozen=True, slots=True)
class SplitFulfillmentDecision:
    action: SplitFulfillmentAction
    outcome: SplitFulfillmentOutcome
    status_message: str
    pull_ranges: tuple[SplitPullRange, ...] = ()
    reusable_range: BlockRange | None = None
    error_message: str | None = None


def split_dataset_facts(
    path: Path,
    validation: BlockDatasetValidationReport,
) -> SplitDatasetFacts:
    return SplitDatasetFacts(
        path=path,
        status=validation.status,
        first_block_number=validation.first_block_number,
        last_block_number=validation.last_block_number,
    )


def plan_history_split_fulfillment(
    target: SplitTarget,
    *,
    existing: SplitDatasetFacts | None,
    staged: SplitDatasetFacts | None,
    staged_matches_target: bool,
) -> SplitFulfillmentDecision:
    if staged is not None:
        invalid = _invalid_staged_decision("history", staged)
        if invalid is not None:
            return invalid
        if staged_matches_target:
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.REUSE_STAGED,
                outcome=_staged_outcome(existing),
                status_message="history reused staged dataset",
            )

    if existing is not None and existing.status == "clean":
        existing_start = _required_first_block(existing)
        existing_end = _required_end_block(existing)
        target_start = target.block_range.start
        target_end = target.block_range.end

        if existing_end == target_end and existing_start <= target_start:
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.REUSE_CACHED,
                outcome=SplitFulfillmentOutcome.REUSED,
                status_message="history reused cached dataset",
            )

        if existing_end == target_end and existing_start > target_start:
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.EXTEND_CACHED,
                outcome=SplitFulfillmentOutcome.EXTENDED,
                status_message="history extending cached dataset",
                pull_ranges=(
                    SplitPullRange(
                        label="history-prefix",
                        block_range=BlockRange(start=target_start, end=existing_start),
                    ),
                ),
            )

    return _full_materialization_decision("history", existing=existing)


def plan_evaluation_split_fulfillment(
    target: SplitTarget,
    *,
    existing: SplitDatasetFacts | None,
    staged: SplitDatasetFacts | None,
    staged_matches_target: bool,
) -> SplitFulfillmentDecision:
    if staged is not None:
        invalid = _invalid_staged_decision("evaluation", staged)
        if invalid is not None:
            return invalid
        if staged_matches_target:
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.REUSE_STAGED,
                outcome=_staged_outcome(existing),
                status_message="evaluation reused staged dataset",
            )

    if existing is not None and existing.status == "clean":
        existing_start = _required_first_block(existing)
        existing_end = _required_end_block(existing)
        target_start = target.block_range.start
        target_end = target.block_range.end

        if existing_start == target_start and existing_end == target_end:
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.REUSE_CACHED,
                outcome=SplitFulfillmentOutcome.REUSED,
                status_message="evaluation reused cached dataset",
            )

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
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
            return SplitFulfillmentDecision(
                action=SplitFulfillmentAction.EXTEND_CACHED,
                outcome=SplitFulfillmentOutcome.EXTENDED,
                status_message="evaluation extending cached dataset",
                pull_ranges=tuple(pull_ranges),
                reusable_range=BlockRange(start=overlap_start, end=overlap_end),
            )

    return _full_materialization_decision("evaluation", existing=existing)


def _invalid_staged_decision(
    kind: str,
    staged: SplitDatasetFacts,
) -> SplitFulfillmentDecision | None:
    if staged.status == "clean":
        return None
    return SplitFulfillmentDecision(
        action=SplitFulfillmentAction.REJECT_INVALID_STAGED,
        outcome=SplitFulfillmentOutcome.REBUILT,
        status_message="",
        error_message=f"Cannot resume invalid staged {kind} dataset",
    )


def _full_materialization_decision(
    kind: str,
    *,
    existing: SplitDatasetFacts | None,
) -> SplitFulfillmentDecision:
    return SplitFulfillmentDecision(
        action=SplitFulfillmentAction.MATERIALIZE_FULL,
        outcome=(
            SplitFulfillmentOutcome.REBUILT
            if existing is not None
            else SplitFulfillmentOutcome.CREATED
        ),
        status_message=f"{kind} downloading",
    )


def _staged_outcome(existing: SplitDatasetFacts | None) -> SplitFulfillmentOutcome:
    return (
        SplitFulfillmentOutcome.REBUILT
        if existing is not None
        else SplitFulfillmentOutcome.CREATED
    )


def _required_first_block(facts: SplitDatasetFacts) -> int:
    if facts.first_block_number is None:
        raise ValueError("validated dataset is missing the first block number")
    return facts.first_block_number


def _required_end_block(facts: SplitDatasetFacts) -> int:
    if facts.last_block_number is None:
        raise ValueError("validated dataset is missing the last block number")
    return facts.last_block_number + 1

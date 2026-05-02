from __future__ import annotations

from pathlib import Path

import pytest

from spice.acquisition import BlockRange, TimestampRange
from spice.corpus.split_fulfillment_plan import (
    SplitDatasetFacts,
    SplitFulfillmentAction,
    SplitFulfillmentOutcome,
    SplitTarget,
    plan_evaluation_split_fulfillment,
    plan_history_split_fulfillment,
)


def _target(kind: str, start: int, end: int) -> SplitTarget:
    return SplitTarget(
        kind=kind,
        block_range=BlockRange(start=start, end=end),
        window=TimestampRange(start=1_000, end=2_000),
    )


def _facts(start: int, end: int, *, status: str = "clean") -> SplitDatasetFacts:
    return SplitDatasetFacts(
        path=Path(f"/dataset/{start}-{end}"),
        status=status,
        first_block_number=start,
        last_block_number=end - 1,
    )


def test_history_decision_reuses_matching_staged_before_cached_dataset() -> None:
    decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=_facts(90, 120),
        staged=_facts(100, 120),
        staged_matches_target=True,
    )

    assert decision.action is SplitFulfillmentAction.REUSE_STAGED
    assert decision.outcome is SplitFulfillmentOutcome.REBUILT
    assert decision.status_message == "history reused staged dataset"


def test_history_decision_rejects_invalid_staged_dataset() -> None:
    decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=None,
        staged=_facts(100, 110, status="error"),
        staged_matches_target=False,
    )

    assert decision.action is SplitFulfillmentAction.REJECT_INVALID_STAGED
    assert decision.error_message == "Cannot resume invalid staged history dataset"


def test_history_decision_reuses_cached_superset_ending_at_boundary() -> None:
    decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=_facts(80, 120),
        staged=None,
        staged_matches_target=False,
    )

    assert decision.action is SplitFulfillmentAction.REUSE_CACHED
    assert decision.outcome is SplitFulfillmentOutcome.REUSED
    assert decision.pull_ranges == ()


def test_history_decision_extends_cached_history_prefix() -> None:
    decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=_facts(110, 120),
        staged=None,
        staged_matches_target=False,
    )

    assert decision.action is SplitFulfillmentAction.EXTEND_CACHED
    assert decision.outcome is SplitFulfillmentOutcome.EXTENDED
    assert [(item.label, item.block_range) for item in decision.pull_ranges] == [
        ("history-prefix", BlockRange(start=100, end=110))
    ]


def test_history_decision_creates_or_rebuilds_full_target() -> None:
    create_decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=None,
        staged=None,
        staged_matches_target=False,
    )
    rebuild_decision = plan_history_split_fulfillment(
        _target("history", 100, 120),
        existing=_facts(130, 140),
        staged=None,
        staged_matches_target=False,
    )

    assert create_decision.action is SplitFulfillmentAction.MATERIALIZE_FULL
    assert create_decision.outcome is SplitFulfillmentOutcome.CREATED
    assert rebuild_decision.action is SplitFulfillmentAction.MATERIALIZE_FULL
    assert rebuild_decision.outcome is SplitFulfillmentOutcome.REBUILT


def test_evaluation_decision_requires_exact_cached_reuse() -> None:
    exact = plan_evaluation_split_fulfillment(
        _target("evaluation", 100, 120),
        existing=_facts(100, 120),
        staged=None,
        staged_matches_target=False,
    )
    superset = plan_evaluation_split_fulfillment(
        _target("evaluation", 100, 120),
        existing=_facts(90, 130),
        staged=None,
        staged_matches_target=False,
    )

    assert exact.action is SplitFulfillmentAction.REUSE_CACHED
    assert superset.action is SplitFulfillmentAction.EXTEND_CACHED
    assert superset.reusable_range == BlockRange(start=100, end=120)
    assert superset.pull_ranges == ()


def test_evaluation_decision_extends_middle_overlap_with_prefix_and_suffix() -> None:
    decision = plan_evaluation_split_fulfillment(
        _target("evaluation", 100, 120),
        existing=_facts(105, 115),
        staged=None,
        staged_matches_target=False,
    )

    assert decision.action is SplitFulfillmentAction.EXTEND_CACHED
    assert decision.reusable_range == BlockRange(start=105, end=115)
    assert [(item.label, item.block_range) for item in decision.pull_ranges] == [
        ("evaluation-prefix", BlockRange(start=100, end=105)),
        ("evaluation-suffix", BlockRange(start=115, end=120)),
    ]


def test_evaluation_decision_rebuilds_when_existing_has_no_overlap() -> None:
    decision = plan_evaluation_split_fulfillment(
        _target("evaluation", 100, 120),
        existing=_facts(130, 150),
        staged=None,
        staged_matches_target=False,
    )

    assert decision.action is SplitFulfillmentAction.MATERIALIZE_FULL
    assert decision.outcome is SplitFulfillmentOutcome.REBUILT


def test_decision_requires_ranges_on_clean_facts() -> None:
    facts = SplitDatasetFacts(
        path=Path("/dataset/missing"),
        status="clean",
        first_block_number=None,
        last_block_number=None,
    )

    with pytest.raises(ValueError, match="first block"):
        plan_history_split_fulfillment(
            _target("history", 100, 120),
            existing=facts,
            staged=None,
            staged_matches_target=False,
        )

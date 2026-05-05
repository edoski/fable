from __future__ import annotations

from pathlib import Path

from spice.acquisition import BlockRange, TimestampRange
from spice.corpus.split_materialization._policy import (
    CorpusSplitOutcome,
    ExtendEvaluationCommittedSplitPlan,
    ExtendHistoryCommittedSplitPlan,
    MaterializeFullSplitPlan,
    RejectInvalidStagedSplitPlan,
    ReuseCommittedSplitPlan,
    SplitDatasetCandidate,
    SplitDatasetFacts,
    SplitPullRange,
    SplitTarget,
    plan_evaluation_split_materialization,
    plan_history_split_materialization,
)
from spice.corpus.validation import BlockDatasetValidationReport


def _candidate(
    tmp_path: Path,
    *,
    status: str = "clean",
    start: int | None = 100,
    last: int | None = 109,
) -> SplitDatasetCandidate:
    path = tmp_path / f"{status}_{start}_{last}"
    return SplitDatasetCandidate(
        path=path,
        validation=BlockDatasetValidationReport(
            dataset_path=path,
            status="clean" if status == "clean" else "error",
            first_block_number=start,
            last_block_number=last,
        ),
        facts=SplitDatasetFacts(
            status=status,
            first_block_number=start,
            last_block_number=last,
        ),
        file_count=1,
    )


def _target(start: int, end: int) -> SplitTarget:
    return SplitTarget(
        kind="history",
        block_range=BlockRange(start=start, end=end),
        window=TimestampRange(start=1_000, end=2_000),
    )


def _accept_target(_: BlockDatasetValidationReport, __: Path) -> None:
    return None


def test_history_policy_reuses_committed_superset(tmp_path: Path) -> None:
    plan = plan_history_split_materialization(
        _target(102, 110),
        existing=_candidate(tmp_path, start=100, last=109),
        staged=None,
        validate_target=_accept_target,
    )

    assert isinstance(plan, ReuseCommittedSplitPlan)
    assert plan.dataset.path == tmp_path / "clean_100_109"
    assert plan.validation.first_block_number == 100


def test_history_policy_executes_prefix_extension(tmp_path: Path) -> None:
    plan = plan_history_split_materialization(
        _target(96, 110),
        existing=_candidate(tmp_path, start=100, last=109),
        staged=None,
        validate_target=_accept_target,
    )

    assert isinstance(plan, ExtendHistoryCommittedSplitPlan)
    assert plan.prefix == SplitPullRange(
        label="history-prefix",
        block_range=BlockRange(start=96, end=100),
    )


def test_evaluation_policy_executes_overlap_extension(tmp_path: Path) -> None:
    plan = plan_evaluation_split_materialization(
        _target(98, 112),
        existing=_candidate(tmp_path, start=100, last=109),
        staged=None,
        validate_target=_accept_target,
        reusable_range_matches_target_window=lambda *_: True,
    )

    assert isinstance(plan, ExtendEvaluationCommittedSplitPlan)
    assert plan.reusable_range == BlockRange(start=100, end=110)
    assert plan.pull_ranges == (
        SplitPullRange(label="evaluation-prefix", block_range=BlockRange(start=98, end=100)),
        SplitPullRange(label="evaluation-suffix", block_range=BlockRange(start=110, end=112)),
    )


def test_policy_rejects_invalid_staged_dataset(tmp_path: Path) -> None:
    plan = plan_evaluation_split_materialization(
        _target(100, 110),
        existing=None,
        staged=_candidate(tmp_path, status="error", start=None, last=None),
        validate_target=_accept_target,
        reusable_range_matches_target_window=lambda *_: True,
    )

    assert isinstance(plan, RejectInvalidStagedSplitPlan)
    assert plan.dataset.file_count == 1
    assert "Cannot resume invalid staged evaluation dataset" == plan.error_message


def test_evaluation_policy_rebuilds_when_overlap_window_is_invalid(tmp_path: Path) -> None:
    plan = plan_evaluation_split_materialization(
        _target(98, 112),
        existing=_candidate(tmp_path, start=100, last=109),
        staged=None,
        validate_target=_accept_target,
        reusable_range_matches_target_window=lambda *_: False,
    )

    assert isinstance(plan, MaterializeFullSplitPlan)
    assert plan.outcome is CorpusSplitOutcome.REBUILT

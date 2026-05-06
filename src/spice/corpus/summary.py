"""Compact acquisition workflow field builders."""

from __future__ import annotations

from ..acquisition import BlockPullPlan
from ..config.models import AcquireConfig
from .split_materialization import CorpusSplitOutcome


def acquire_dry_run_fields(
    config: AcquireConfig,
    *,
    history_window_seconds: int,
    history_plan: BlockPullPlan,
    evaluation_plan: BlockPullPlan,
) -> list[tuple[str, str]]:
    del config
    return [
        ("history_window", f"{history_window_seconds}s"),
        ("history_blocks", str(history_plan.block_range.count)),
        ("evaluation_blocks", str(evaluation_plan.block_range.count)),
    ]


def acquisition_result_fields(
    *,
    history_outcome: CorpusSplitOutcome,
    history_row_count: int,
    evaluation_outcome: CorpusSplitOutcome,
    evaluation_row_count: int,
) -> list[tuple[str, str]]:
    return [
        ("history", history_outcome.value),
        ("history_blocks", str(history_row_count)),
        ("evaluation", evaluation_outcome.value),
        ("evaluation_blocks", str(evaluation_row_count)),
    ]

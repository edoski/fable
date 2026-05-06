from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import cast

import pytest

from spice.acquisition import BlockPullPlan, BlockRange, TimestampRange
from spice.config import AcquireConfig, WorkflowTask, coerce_features_config
from spice.config.groups import load_named_group_payload
from spice.corpus.planning import (
    HISTORY_REFILL_ATTEMPT_LIMIT,
    CorpusCapabilityPlanningSpec,
    build_corpus_capability_planning_context,
)
from spice.corpus.validation import BlockDatasetValidationReport


def _load_acquire_config(
    load_workflow_config,
    tmp_path,
    *,
    override: dict[str, object] | None = None,
) -> AcquireConfig:
    return cast(
        AcquireConfig,
        load_workflow_config(
            WorkflowTask.ACQUIRE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )


def _planning_spec(config: AcquireConfig) -> CorpusCapabilityPlanningSpec:
    return CorpusCapabilityPlanningSpec(
        features=config.features,
        problem=config.problem,
        chain_runtime=config.chain.runtime,
        history_window_end_timestamp=config.history_window_end_timestamp,
        evaluation_window_start_timestamp=config.evaluation_window_start_timestamp,
        evaluation_window_end_timestamp=config.evaluation_window_end_timestamp,
    )


def _plan_for_window(window: TimestampRange, *, start_block: int) -> BlockPullPlan:
    row_count = max(1, math.ceil((window.end - window.start) / 12))
    return BlockPullPlan(
        window=window,
        block_range=BlockRange(start=start_block, end=start_block + row_count)
    )


class _PlanningSource:
    def __init__(self, evaluation_window: TimestampRange) -> None:
        self.evaluation_window = evaluation_window
        self.planned_windows: list[TimestampRange] = []

    async def close(self) -> None:
        return None

    async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
        del sample_size
        return 12.0

    async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
        self.planned_windows.append(window)
        return _plan_for_window(
            window,
            start_block=10_000 if window == self.evaluation_window else 100,
        )

    def plan_block_range(
        self,
        block_range: BlockRange,
        *,
        window: TimestampRange,
    ) -> BlockPullPlan:
        return BlockPullPlan(
            window=window,
            block_range=block_range
        )

    async def get_block_rows(self, start: int, end: int):
        del start, end
        raise AssertionError("planning must not fetch rows")


def test_initial_capability_planning_computes_evaluation_and_history_windows(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    context = build_corpus_capability_planning_context(_planning_spec(config))
    evaluation_window = TimestampRange(
        start=config.evaluation_window_start_timestamp,
        end=config.evaluation_window_end_timestamp,
    )
    source = _PlanningSource(evaluation_window)

    plan = asyncio.run(context.initial_plan(source))

    expected_history_seconds = math.ceil(
        context.problem_contract.initial_history_window_seconds(12.0) * 1.1
    )
    assert plan.evaluation_plan.window == evaluation_window
    assert plan.requested_history_window_seconds == expected_history_seconds
    assert plan.history_plan.window == TimestampRange(
        start=config.history_window_end_timestamp - expected_history_seconds,
        end=config.history_window_end_timestamp,
    )
    assert source.planned_windows == [evaluation_window, plan.history_plan.window]


def test_capability_planning_derives_priority_fee_source_requirements(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    priority_config = config.model_copy(
        update={
            "features": coerce_features_config(
                load_named_group_payload("core_fee_dynamics_with_priority_fee", "features")
            )
        }
    )

    baseline_context = build_corpus_capability_planning_context(_planning_spec(config))
    priority_context = build_corpus_capability_planning_context(
        _planning_spec(priority_config)
    )

    assert baseline_context.source_requirements.required_columns >= {
        "block_number",
        "timestamp",
        "chain_id",
        "base_fee_per_gas",
        "gas_used",
        "gas_limit",
        "tx_count",
    }
    assert "priority_fee_percentiles" not in (
        baseline_context.source_requirements.optional_enrichments
    )
    assert "priority_fee_percentiles" in (
        priority_context.source_requirements.optional_enrichments
    )
    assert priority_context.source_requirements.required_columns >= {
        "priority_fee_p10",
        "priority_fee_p50",
        "priority_fee_p90",
        "priority_fee_spread",
    }
    assert priority_context.source_requirements.temporal_unit == "block"
    assert priority_context.source_requirements.ordering_key == "block_number"
    assert priority_context.source_requirements.partition_key == "chain_id"


def test_capability_planning_builds_refill_decision(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(sample_count=4),
    )
    context = build_corpus_capability_planning_context(_planning_spec(config))
    evaluation_window = TimestampRange(
        start=config.evaluation_window_start_timestamp,
        end=config.evaluation_window_end_timestamp,
    )
    source = _PlanningSource(evaluation_window)
    validation = BlockDatasetValidationReport(
        dataset_path=Path("history"),
        row_count=51,
        first_timestamp=config.history_window_end_timestamp - 600,
        last_timestamp=config.history_window_end_timestamp,
    )

    refill = asyncio.run(
        context.plan_history_refill(
            block_source=source,
            validation=validation,
            resolved_capability_samples=1,
            requested_history_window_seconds=600,
        )
    )

    assert refill is not None
    assert refill.requested_history_window_seconds == 640
    assert refill.history_plan.window == TimestampRange(
        start=config.history_window_end_timestamp - 640,
        end=config.history_window_end_timestamp,
    )
    assert refill.status_message == "history refilling samples=1/4"


def test_capability_planning_reports_bounded_refill_failure(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(sample_count=4),
    )
    context = build_corpus_capability_planning_context(_planning_spec(config))

    with pytest.raises(RuntimeError, match=f"refill_attempts={HISTORY_REFILL_ATTEMPT_LIMIT}"):
        context.ensure_sufficient_history_samples(2)

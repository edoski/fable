from __future__ import annotations

import asyncio
import math
from io import StringIO
from typing import cast

import pytest

import spice.workflows.acquire as acquire_workflow
from spice.acquisition.rpc import (
    BlockPullPlan,
    BlockRange,
    TimestampRange,
)
from spice.config import AcquireConfig, WorkflowTask
from spice.core.reporting import Reporter
from spice.storage.corpus import list_acquire_runs, load_dataset_manifest
from spice.storage.layout import resolve_workflow_paths
from spice.storage.roots import DatasetSelector, list_dataset_records
from spice.workflows.acquire import run as run_acquire
from tests.dataset_helpers import make_block_rows


def _load_test_acquire_config(
    load_workflow_config,
    tmp_path,
    *,
    override: dict[str, object] | None = None,
    chain: str | None = None,
) -> AcquireConfig:
    return cast(
        AcquireConfig,
        load_workflow_config(
            WorkflowTask.ACQUIRE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
            chain=chain,
        ),
    )


def _plan_for_window(
    window: TimestampRange,
    *,
    start_block: int,
    chunk_size: int,
    block_interval_seconds: int = 12,
    expected_rows: int | None = None,
) -> BlockPullPlan:
    row_count = (
        expected_rows
        if expected_rows is not None
        else max(1, math.ceil((window.end - window.start) / block_interval_seconds))
    )
    return BlockPullPlan(
        window=window,
        block_range=BlockRange(start=start_block, end=start_block + row_count),
        expected_rows=row_count,
    )


def test_acquire_workflow_writes_canonical_corpus_and_metadata(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    paths = resolve_workflow_paths(config)
    evaluation_plan = _plan_for_window(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        start_block=10_000,
        expected_rows=32,
        chunk_size=config.acquisition.chunk_size,
    )
    history_windows: list[TimestampRange] = []

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint
            self.chain = chain
            self._planned_windows: list[BlockPullPlan] = []

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            if window == evaluation_plan.window:
                plan = evaluation_plan
            else:
                history_windows.append(window)
                plan = _plan_for_window(
                    window,
                    start_block=100,
                    chunk_size=config.acquisition.chunk_size,
                )
            self._planned_windows.append(plan)
            return plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
        ) -> BlockPullPlan:
            plan = BlockPullPlan(
                window,
                block_range=block_range,
                expected_rows=block_range.count,
            )
            self._planned_windows.append(plan)
            return plan

        async def get_block_rows(self, start: int, end: int):
            first_block = start
            for plan in reversed(self._planned_windows):
                if plan.block_range.start <= first_block < plan.block_range.end:
                    return make_block_rows(
                        end - start,
                        start_block=first_block,
                        start_timestamp=(
                            plan.window.start + (first_block - plan.block_range.start) * 12
                        ),
                        chain_id=config.chain.runtime.chain_id,
                        block_interval_seconds=12,
                    )
            raise AssertionError(f"missing plan for block {first_block}")

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    run_acquire(config)

    summary = load_dataset_manifest(paths.corpus_state_db)
    runs = list_acquire_runs(paths.corpus_state_db)
    assert paths.corpus_state_db.is_file()
    assert summary.validation.evaluation.rows == evaluation_plan.expected_rows
    assert summary.dataset.id == paths.corpus_id
    assert summary.dataset.name == config.dataset.name
    assert summary.chain.name == config.chain.name
    assert len(runs) == 1
    assert runs[0].facts.resolved_capability_samples >= config.problem.sample_count
    assert paths.history_dir.is_dir()
    assert paths.evaluation_dir.is_dir()
    datasets = list_dataset_records(
        paths.output_root,
        selector=DatasetSelector(
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
        ),
    )
    assert len(datasets) == 1
    assert datasets[0].dataset_id == paths.corpus_id


def test_acquire_cancellation_during_planning_logs_warning(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    paths = resolve_workflow_paths(config)
    output = StringIO()
    errors = StringIO()
    reporter = Reporter(stream=output, error_stream=errors)

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint, chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            await asyncio.sleep(0.2)
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            await asyncio.sleep(0.2)
            return _plan_for_window(
                window,
                start_block=100,
                chunk_size=config.acquisition.chunk_size,
            )

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    async def _exercise() -> None:
        problem = asyncio.create_task(acquire_workflow._run_async(config, reporter=reporter))
        await asyncio.sleep(0.05)
        problem.cancel()
        with pytest.raises(asyncio.CancelledError):
            await problem

    asyncio.run(_exercise())

    rendered = output.getvalue()
    assert "acquire dataset=" in rendered
    assert (
        errors.getvalue()
        == "warning: acquire cancelled; partial staging preserved for resume\n"
    )
    assert paths.corpus_state_db.exists() is False


def test_acquire_failure_preserves_staging_and_rerun_resumes(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    override = acquire_override()
    override["acquisition"] = {
        "dry_run": False,
        "chunk_size": 4,
        "rpc": {
            "batch_size": 4,
            "concurrency": 1,
            "min_batch_size": 4,
            "concurrency_rungs": [1],
        },
    }
    config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=override,
    )
    paths = resolve_workflow_paths(config)
    stage_root = paths.corpus_root.parent / f".{paths.corpus_id}.acquire-staging"
    evaluation_plan = _plan_for_window(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        start_block=10_000,
        expected_rows=4,
        chunk_size=config.acquisition.chunk_size,
    )
    history_plan = _plan_for_window(
        TimestampRange(
            start=config.history_window_end_timestamp - 120,
            end=config.history_window_end_timestamp,
        ),
        start_block=100,
        expected_rows=10,
        chunk_size=config.acquisition.chunk_size,
    )
    requested_ranges: list[tuple[int, int]] = []
    fail_history_tail = True

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint
            self.chain = chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            return evaluation_plan if window == evaluation_plan.window else history_plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
            )

        async def get_block_rows(self, start: int, end: int):
            requested_ranges.append((start, end))
            if fail_history_tail and start == 108:
                raise TimeoutError("synthetic transient failure")
            plan = evaluation_plan if start >= evaluation_plan.block_range.start else history_plan
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=plan.window.start + (start - plan.block_range.start) * 12,
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)
    monkeypatch.setattr(
        "spice.workflows.acquire._count_valid_history_samples",
        lambda **_: config.problem.sample_count,
    )

    with pytest.raises(RuntimeError, match="exceeded 8 transient retry attempts"):
        run_acquire(config)

    assert stage_root.is_dir()
    assert len(list(stage_root.rglob("*.parquet"))) == 2
    assert paths.corpus_state_db.exists() is False
    assert requested_ranges.count((100, 104)) == 1
    assert requested_ranges.count((104, 108)) == 1

    fail_history_tail = False
    run_acquire(config)

    assert stage_root.exists() is False
    assert paths.corpus_state_db.is_file()
    assert paths.history_dir.is_dir()
    assert paths.evaluation_dir.is_dir()
    assert requested_ranges.count((100, 104)) == 1
    assert requested_ranges.count((104, 108)) == 1
    assert (108, 110) in requested_ranges


def test_acquire_dry_run_emits_compact_output(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    config.acquisition.dry_run = True
    paths = resolve_workflow_paths(config)
    output = StringIO()
    reporter = Reporter(stream=output)

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint, chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            return _plan_for_window(
                window,
                start_block=100,
                chunk_size=config.acquisition.chunk_size,
            )

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    run_acquire(config, reporter=reporter)

    rendered = output.getvalue()
    assert "acquire dry_run" in rendered
    assert not paths.corpus_state_db.exists()


def _exercise_short_history_refill(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
    *,
    final_sample_count: int | None = None,
    expect_error: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[TimestampRange], int]:
    override = acquire_override()
    override["acquisition"] = {
        "chunk_size": 128,
        "rpc": {
            "batch_size": 128,
            "concurrency": 1,
            "min_batch_size": 128,
            "concurrency_rungs": [1],
        },
    }
    config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=override,
    )
    evaluation_plan = _plan_for_window(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        start_block=10_000,
        expected_rows=32,
        chunk_size=config.acquisition.chunk_size,
    )
    history_plans = [
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 50 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=1_000,
            expected_rows=50,
            chunk_size=config.acquisition.chunk_size,
        ),
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 100 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=950,
            expected_rows=100,
            chunk_size=config.acquisition.chunk_size,
        ),
    ]
    partial_ranges: list[tuple[int, int]] = []
    requested_ranges: list[tuple[int, int]] = []
    history_windows: list[TimestampRange] = []
    resolved_capability_samples = iter(
        [1, config.problem.sample_count if final_sample_count is None else final_sample_count]
    )
    history_plan_calls = 0

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint
            self.chain = chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            if window == evaluation_plan.window:
                return evaluation_plan
            nonlocal history_plan_calls
            template = history_plans[history_plan_calls]
            history_plan_calls += 1
            history_windows.append(window)
            return BlockPullPlan(
                window=window,
                block_range=template.block_range,
                expected_rows=template.expected_rows,
            )

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
        ) -> BlockPullPlan:
            partial_ranges.append((block_range.start, block_range.end))
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
            )

        async def get_block_rows(self, start: int, end: int):
            first_block = start
            end_block = end
            requested_ranges.append((first_block, end_block))
            if first_block >= evaluation_plan.block_range.start:
                plan = evaluation_plan
            else:
                plan = (
                    history_plans[0]
                    if first_block >= history_plans[0].block_range.start
                    else history_plans[1]
                )
            return make_block_rows(
                end - start,
                start_block=first_block,
                start_timestamp=plan.window.start + (first_block - plan.block_range.start) * 12,
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)
    monkeypatch.setattr(
        "spice.workflows.acquire._count_valid_history_samples",
        lambda **_: next(resolved_capability_samples),
    )

    if expect_error:
        with pytest.raises(RuntimeError, match="under-requested capability samples"):
            run_acquire(config)
    else:
        run_acquire(config)
    return partial_ranges, requested_ranges, history_windows, history_plan_calls


def test_acquire_workflow_refills_missing_history_prefix_once(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    partial_ranges, requested_ranges, history_windows, history_plan_calls = (
        _exercise_short_history_refill(
            tmp_path,
            monkeypatch,
            load_workflow_config,
            acquire_override,
        )
    )

    assert partial_ranges == [(950, 1_000)]
    assert (1_000, 1_050) in requested_ranges
    assert (950, 1_000) in requested_ranges
    assert (950, 1_050) not in requested_ranges


def test_acquire_workflow_fails_after_one_short_refill(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    _exercise_short_history_refill(
        tmp_path,
        monkeypatch,
        load_workflow_config,
        acquire_override,
        final_sample_count=2,
        expect_error=True,
    )

from __future__ import annotations

import asyncio
from concurrent.futures.thread import _threads_queues
from pathlib import Path

from spice.acquisition.provider import ManagedAsyncHTTPProvider
from spice.acquisition.rpc import BlockPullPlan, BlockRange, TimestampRange
from spice.core.console import NullReporter
from spice.state.dataset import list_acquire_runs, load_dataset_summary
from spice.workflows.acquire import _DaemonThreadPoolExecutor
from spice.workflows.acquire import run as run_acquire
from tests.support import (
    acquire_override,
    load_test_acquire_config,
    make_block_rows,
    required_history_blocks,
    write_dataset_dir,
)


def test_acquire_workflow_writes_canonical_dataset_and_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    required_blocks = required_history_blocks(config)
    block_time_seconds = int(round(config.chain.runtime.block_time_seconds))
    history_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp - required_blocks * block_time_seconds,
            end=config.evaluation_window_start_timestamp,
        ),
        block_range=BlockRange(start=100, end=100 + required_blocks),
        expected_rows=required_blocks,
        expected_files=1,
    )
    evaluation_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        block_range=BlockRange(start=10_000, end=10_032),
        expected_rows=32,
        expected_files=1,
    )

    class FakeAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            del end_timestamp, required_history_blocks, chunk_size
            return history_plan

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            return evaluation_plan if window == evaluation_plan.window else history_plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=plan.window.start,
                chain_id=config.chain.runtime.chain_id,
                block_time_seconds=block_time_seconds,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeAcquireClient)

    run_acquire(config, reporter=NullReporter())

    summary = load_dataset_summary(config.paths.dataset_state_db)
    runs = list_acquire_runs(config.paths.dataset_state_db)
    assert config.paths.dataset_state_db.is_file()
    assert summary.validation.history.rows == required_blocks
    assert summary.validation.evaluation.rows == evaluation_plan.expected_rows
    assert summary.provider.name == "publicnode"
    assert len(runs) == 1
    assert runs[0]["task_id"] == config.task.id
    assert runs[0]["feature_set_id"] == config.feature_set.id
    assert runs[0]["required_history_blocks"] == required_blocks
    assert config.paths.history_dir.is_dir()
    assert config.paths.evaluation_dir.is_dir()


def test_acquire_run_swallows_keyboard_interrupt(tmp_path, monkeypatch) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())

    def _raise_keyboard_interrupt(coro) -> None:
        coro.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "spice.workflows.acquire._run_async_interruptibly",
        _raise_keyboard_interrupt,
    )

    run_acquire(config, reporter=NullReporter())


def test_acquire_executor_threads_skip_python_exit_registry() -> None:
    executor = _DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="spice-test")
    try:
        future = executor.submit(lambda: None)
        future.result(timeout=1)
        assert executor._threads
        assert all(thread not in _threads_queues for thread in executor._threads)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def test_managed_async_http_provider_disconnect_closes_managed_session() -> None:
    provider = ManagedAsyncHTTPProvider("http://localhost:8545")

    async def _exercise() -> tuple[bool, bool]:
        session = await provider._request_session_manager.async_cache_and_return_session(
            "http://localhost:8545"
        )
        before = session.closed
        await provider.disconnect()
        return before, session.closed

    before_closed, after_closed = asyncio.run(_exercise())
    assert before_closed is False
    assert after_closed is True

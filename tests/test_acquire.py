from __future__ import annotations

import asyncio
import math
from concurrent.futures.thread import _threads_queues
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import spice.workflows.acquire as acquire_workflow
from spice.acquisition.provider import ManagedAsyncHTTPProvider
from spice.acquisition.rpc import (
    BlockPullPlan,
    BlockRange,
    RpcController,
    TimestampRange,
    Web3BlockClient,
)
from spice.core.console import NullReporter, PlainReporter
from spice.planning.contracts import resolve_task_contract
from spice.state.catalog import list_dataset_records
from spice.state.dataset import list_acquire_runs, load_dataset_summary
from spice.workflows.acquire import _DaemonThreadPoolExecutor
from spice.workflows.acquire import run as run_acquire
from tests.support import (
    acquire_override,
    load_test_acquire_config,
    make_block_rows,
    write_dataset_dir,
)


class CaptureReporter(NullReporter):
    def __init__(self) -> None:
        self.messages: list[str | None] = []

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> int:
        del name, total, unit
        return 1

    def update_task(
        self,
        task_id: int,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
    ) -> None:
        del task_id, completed, advance
        self.messages.append(message)


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
        expected_files=max(1, math.ceil(row_count / chunk_size)),
    )


def test_acquire_workflow_writes_canonical_corpus_and_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    contract = resolve_task_contract(
        task=config.task,
        feature_set=config.feature_set,
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

    class FakeAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            if window == evaluation_plan.window:
                return evaluation_plan
            return _plan_for_window(
                window,
                start_block=100,
                chunk_size=chunk_size,
            )

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
                expected_files=max(1, math.ceil(block_range.count / chunk_size)),
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
                block_interval_seconds=12,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeAcquireClient)

    run_acquire(config, reporter=NullReporter())

    summary = load_dataset_summary(config.paths.corpus_state_db)
    runs = list_acquire_runs(config.paths.corpus_state_db)
    assert config.paths.corpus_state_db.is_file()
    assert summary.validation.evaluation.rows == evaluation_plan.expected_rows
    assert summary.provider.name == "publicnode"
    assert len(runs) == 1
    assert runs[0]["task_id"] == config.task.id
    assert runs[0]["feature_set_id"] == config.feature_set.id
    assert runs[0]["feature_history_seconds"] == contract.feature_history_seconds
    assert runs[0]["required_history_seconds"] == contract.required_history_seconds
    assert runs[0]["valid_anchor_samples"] >= config.task.sample_count
    assert runs[0]["acquired_history_window_seconds"] >= contract.required_history_seconds
    assert config.paths.history_dir.is_dir()
    assert config.paths.evaluation_dir.is_dir()
    datasets = list_dataset_records(
        config.paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
    )
    assert len(datasets) == 1
    assert datasets[0].dataset_id == config.paths.corpus_id


def test_acquire_cancellation_during_planning_logs_warning(tmp_path, monkeypatch) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    output = StringIO()
    reporter = PlainReporter(console=Console(file=output, force_terminal=False, width=160))

    class FakeAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider, chain

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            await asyncio.sleep(0.2)
            return 12.0

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            await asyncio.sleep(0.2)
            return _plan_for_window(
                window,
                start_block=100,
                chunk_size=config.acquisition.chunk_size,
            )

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeAcquireClient)

    async def _exercise() -> None:
        task = asyncio.create_task(acquire_workflow._run_async(config, reporter=reporter))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())

    rendered = output.getvalue()
    assert "history [planning] - resolving window" in rendered
    assert "evaluation [planning] - resolving window" in rendered
    assert "warning: acquire cancelled; partial download removed" in rendered
    assert config.paths.corpus_state_db.exists() is False


def test_pull_block_range_emits_structured_progress_messages(tmp_path) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    reporter = CaptureReporter()
    controller = RpcController(
        configured_batch_size=16,
        min_batch_size=8,
        concurrency_rungs=(8,),
        configured_concurrency=8,
    )
    plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_start_timestamp + 16 * 12,
        ),
        block_range=BlockRange(start=100, end=116),
        expected_rows=16,
        expected_files=1,
    )

    class FakeClient(Web3BlockClient):
        def __post_init__(self) -> None:
            self._calls = 0

        async def close(self) -> None:
            return None

        async def get_block_rows(self, block_numbers: list[int]):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("response too large")
            return make_block_rows(
                len(block_numbers),
                start_block=block_numbers[0],
                start_timestamp=(
                    plan.window.start
                    + (block_numbers[0] - plan.block_range.start) * 12
                ),
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    client = FakeClient(config.provider, config.chain)

    asyncio.run(
        client.pull_block_range(
            tmp_path / "history",
            plan=plan,
            chunk_size=64,
            rpc_controller=controller,
            reporter=reporter,
        )
    )

    assert reporter.messages[0] == "batch=16 conc=8"
    assert "oversize backoff batch=8 conc=8" in reporter.messages
    assert reporter.messages.count("batch=8 conc=8") == 2


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

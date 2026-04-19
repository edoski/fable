from __future__ import annotations

import asyncio
import math
from concurrent.futures.thread import _threads_queues
from io import StringIO

import aiohttp
import pytest
from web3.providers.rpc.utils import ExceptionRetryConfiguration

import spice.workflows.acquire as acquire_workflow
from spice.acquisition.provider import ManagedAsyncHTTPProvider
from spice.acquisition.rpc import (
    BlockPullPlan,
    BlockRange,
    BlockRpcClient,
    RpcController,
    TimestampRange,
    pull_block_range,
)
from spice.core.reporting import NullReporter, PlainReporter, StageMetricValue
from spice.features import compile_feature_contract
from spice.storage.catalog import list_dataset_records
from spice.storage.corpus import list_acquire_runs, load_dataset_manifest
from spice.storage.layout import resolve_workflow_paths
from spice.temporal.contracts import compile_problem_contract
from spice.workflows.acquire import _DaemonThreadPoolExecutor
from spice.workflows.acquire import run as run_acquire


class CaptureReporter(NullReporter):
    def __init__(self) -> None:
        self.messages: list[str | None] = []
        self.metrics: list[dict[str, str]] = []
        self.completions: list[int | None] = []

    def start_task(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str | None = None,
        completed: int | None = None,
    ) -> int:
        del name, total, unit, completed
        return 1

    def update_task(
        self,
        problem_id: int,
        *,
        completed: int | None = None,
        advance: int | None = None,
        message: str | None = None,
        metrics: tuple[StageMetricValue, ...] = (),
    ) -> None:
        del problem_id, advance
        self.messages.append(message)
        self.metrics.append({metric.id: metric.value for metric in metrics})
        self.completions.append(completed)


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
    load_test_acquire_config,
    acquire_override,
    make_block_rows,
) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    paths = resolve_workflow_paths(config)
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
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
            self._planned_windows: list[BlockPullPlan] = []

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            plan = (
                evaluation_plan
                if window == evaluation_plan.window
                else _plan_for_window(
                    window,
                    start_block=100,
                    chunk_size=chunk_size,
                )
            )
            self._planned_windows.append(plan)
            return plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            plan = BlockPullPlan(
                window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, math.ceil(block_range.count / chunk_size)),
            )
            self._planned_windows.append(plan)
            return plan

        async def get_block_rows(self, block_numbers: list[int]):
            first_block = block_numbers[0]
            for plan in reversed(self._planned_windows):
                if plan.block_range.start <= first_block < plan.block_range.end:
                    return make_block_rows(
                        len(block_numbers),
                        start_block=first_block,
                        start_timestamp=(
                            plan.window.start + (first_block - plan.block_range.start) * 12
                        ),
                        chain_id=config.chain.runtime.chain_id,
                        block_interval_seconds=12,
                    )
            raise AssertionError(f"missing plan for block {first_block}")

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    run_acquire(config, reporter=NullReporter())

    summary = load_dataset_manifest(paths.corpus_state_db)
    runs = list_acquire_runs(paths.corpus_state_db)
    assert paths.corpus_state_db.is_file()
    assert summary.validation.evaluation.rows == evaluation_plan.expected_rows
    assert summary.semantics.problem.problem_id == config.problem.id
    assert summary.semantics.problem.compiler_id == config.problem.compiler.id
    assert summary.semantics.feature.feature_set_id == config.feature_set.id
    assert summary.semantics.feature.feature_family_id == config.feature_set.family.id
    assert (
        summary.semantics.feature.feature_graph_fingerprint
        == feature_contract.feature_graph_fingerprint
    )
    assert summary.semantics.feature.feature_prerequisites == contract.feature_prerequisites
    assert len(runs) == 1
    assert runs[0].provider.name == "publicnode"
    assert runs[0].facts.required_history_seconds == contract.required_history_seconds
    assert runs[0].facts.valid_anchor_samples >= config.problem.sample_count
    assert runs[0].facts.acquired_history_window_seconds >= contract.required_history_seconds
    assert paths.history_dir.is_dir()
    assert paths.evaluation_dir.is_dir()
    datasets = list_dataset_records(
        paths.catalog_db,
        chain_name=config.chain.name,
        dataset_name=config.dataset.name,
    )
    assert len(datasets) == 1
    assert datasets[0].dataset_id == paths.corpus_id


def test_acquire_cancellation_during_planning_logs_warning(
    tmp_path,
    monkeypatch,
    load_test_acquire_config,
    acquire_override,
) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    paths = resolve_workflow_paths(config)
    output = StringIO()
    reporter = PlainReporter(stream=output)

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

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    async def _exercise() -> None:
        problem = asyncio.create_task(acquire_workflow._run_async(config, reporter=reporter))
        await asyncio.sleep(0.05)
        problem.cancel()
        with pytest.raises(asyncio.CancelledError):
            await problem

    asyncio.run(_exercise())

    rendered = output.getvalue()
    assert "history [planning] - resolving window" in rendered
    assert "evaluation [planning] - resolving window" in rendered
    assert "warning: acquire cancelled; partial download removed" in rendered
    assert paths.corpus_state_db.exists() is False


def test_pull_block_range_emits_structured_progress_messages(
    tmp_path,
    load_test_acquire_config,
    acquire_override,
    make_block_rows,
) -> None:
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

    class FakeClient(BlockRpcClient):
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
                    plan.window.start + (block_numbers[0] - plan.block_range.start) * 12
                ),
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    client = FakeClient(config.provider, config.chain)

    asyncio.run(
        pull_block_range(
            client,
            tmp_path / "history",
            plan=plan,
            chunk_size=64,
            rpc_controller=controller,
            reporter=reporter,
        )
    )

    assert reporter.messages[0] is None
    assert reporter.metrics[0] == {"batch": "16", "conc": "8"}
    assert ("oversize backoff", {"batch": "8", "conc": "8"}) in list(
        zip(reporter.messages, reporter.metrics, strict=True)
    )
    assert reporter.metrics.count({"batch": "8", "conc": "8"}) == 2


def test_pull_block_range_coalesces_simultaneous_success_updates(
    tmp_path,
    load_test_acquire_config,
    acquire_override,
    make_block_rows,
) -> None:
    config = load_test_acquire_config(tmp_path, override=acquire_override())
    reporter = CaptureReporter()
    controller = RpcController(
        configured_batch_size=8,
        min_batch_size=8,
        concurrency_rungs=(2,),
        configured_concurrency=2,
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

    class FakeClient(BlockRpcClient):
        async def close(self) -> None:
            return None

        async def get_block_rows(self, block_numbers: list[int]):
            return make_block_rows(
                len(block_numbers),
                start_block=block_numbers[0],
                start_timestamp=(
                    plan.window.start + (block_numbers[0] - plan.block_range.start) * 12
                ),
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    client = FakeClient(config.provider, config.chain)

    asyncio.run(
        pull_block_range(
            client,
            tmp_path / "history",
            plan=plan,
            chunk_size=64,
            rpc_controller=controller,
            reporter=reporter,
        )
    )

    assert reporter.completions == [0, 16]


def test_acquire_workflow_reuses_temporary_history_between_expansions(
    tmp_path,
    monkeypatch,
    load_test_acquire_config,
    acquire_override,
    make_block_rows,
) -> None:
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
    config = load_test_acquire_config(tmp_path, override=override)
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
    valid_anchor_samples = iter([1, config.problem.sample_count])

    class FakeAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain
            self._history_plan_calls = 0

        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            if window == evaluation_plan.window:
                return evaluation_plan
            plan = history_plans[self._history_plan_calls]
            self._history_plan_calls += 1
            return plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            partial_ranges.append((block_range.start, block_range.end))
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, math.ceil(block_range.count / chunk_size)),
            )

        async def get_block_rows(self, block_numbers: list[int]):
            first_block = block_numbers[0]
            end_block = block_numbers[-1] + 1
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
                len(block_numbers),
                start_block=first_block,
                start_timestamp=plan.window.start + (first_block - plan.block_range.start) * 12,
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)
    monkeypatch.setattr(
        "spice.workflows.acquire._count_valid_history_samples",
        lambda **_: next(valid_anchor_samples),
    )

    run_acquire(config, reporter=NullReporter())

    assert partial_ranges == [(950, 1_000)]
    assert (1_000, 1_050) in requested_ranges
    assert (950, 1_000) in requested_ranges
    assert (950, 1_050) not in requested_ranges


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


def test_managed_async_http_provider_retries_batch_transport_errors(monkeypatch) -> None:
    provider = ManagedAsyncHTTPProvider(
        "http://localhost:8545",
        exception_retry_configuration=ExceptionRetryConfiguration(
            errors=[aiohttp.ClientError],
            retries=2,
            backoff_factor=0.0,
        ),
    )
    attempts = 0

    async def _fake_post_request(endpoint_uri: str, data: bytes, **kwargs: object) -> bytes:
        del endpoint_uri, data, kwargs
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise aiohttp.ClientPayloadError(
                "Response payload is not completed: "
                "<TransferEncodingError: 400, "
                "message='Not enough data to satisfy transfer length header.'>"
            )
        return b'[{"jsonrpc":"2.0","id":1,"result":"ok"}]'

    monkeypatch.setattr(
        provider._request_session_manager,
        "async_make_post_request",
        _fake_post_request,
    )
    monkeypatch.setattr(provider, "encode_batch_rpc_request", lambda requests: b"[]")

    response = asyncio.run(provider.make_batch_request([("eth_test", [])]))

    assert attempts == 2
    assert response == [{"jsonrpc": "2.0", "id": 1, "result": "ok"}]

from __future__ import annotations

import asyncio
import math
from io import StringIO
from typing import cast

import aiohttp
import pytest
from web3.providers.rpc.utils import ExceptionRetryConfiguration

import spice.workflows.acquire as acquire_workflow
from spice.acquisition.rpc import (
    BlockPullPlan,
    BlockRange,
    TimestampRange,
)
from spice.acquisition.rpc.transport import ManagedAsyncHTTPProvider
from spice.config import AcquireConfig, WorkflowTask
from spice.core.reporting import Reporter
from spice.features import compile_feature_contract
from spice.storage.catalog import list_dataset_records
from spice.storage.corpus import list_acquire_runs, load_dataset_manifest
from spice.storage.layout import resolve_workflow_paths
from spice.temporal.contracts import compile_problem_contract
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
            preset="icdcs_2026",
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
        expected_files=max(1, math.ceil(row_count / chunk_size)),
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
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    expected_history_window_seconds = math.ceil(
        contract.initial_history_window_seconds(12.0) * 1.10
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

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            if window == evaluation_plan.window:
                plan = evaluation_plan
            else:
                history_windows.append(window)
                plan = _plan_for_window(
                    window,
                    start_block=100,
                    chunk_size=chunk_size,
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

    run_acquire(config)

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
    assert runs[0].provider.name == config.rpc_endpoint.provider_name
    assert runs[0].provider.reference == config.rpc_endpoint.reference
    assert runs[0].facts.requested_history_window_seconds == expected_history_window_seconds
    assert runs[0].facts.resolved_capability_samples >= config.problem.sample_count
    assert len(history_windows) == 1
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
    reporter = Reporter(stream=output)

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain) -> None:
            del rpc_endpoint, chain

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
    assert "acquire dataset=" in rendered
    assert "warning: acquire cancelled; partial download removed" in rendered
    assert paths.corpus_state_db.exists() is False


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

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            return _plan_for_window(window, start_block=100, chunk_size=chunk_size)

    monkeypatch.setattr("spice.workflows.acquire.BlockRpcClient", FakeAcquireClient)

    run_acquire(config, reporter=reporter)

    rendered = output.getvalue()
    assert "acquire dataset=" in rendered
    assert "acquire dry_run history_window=" in rendered
    assert "history_blocks=" in rendered
    assert "evaluation_blocks=" in rendered


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

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
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
                expected_files=template.expected_files,
            )

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
    assert history_plan_calls == 2
    assert len(history_windows) == 2
    assert (history_windows[0].end - history_windows[1].start) < 2 * (
        history_windows[0].end - history_windows[0].start
    )


def test_acquire_workflow_fails_after_one_short_refill(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    _, _, _, history_plan_calls = _exercise_short_history_refill(
        tmp_path,
        monkeypatch,
        load_workflow_config,
        acquire_override,
        final_sample_count=2,
        expect_error=True,
    )

    assert history_plan_calls == 2


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

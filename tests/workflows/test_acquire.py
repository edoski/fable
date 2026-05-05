from __future__ import annotations

import math
from io import StringIO
from typing import cast

import pytest

from spice.acquisition import (
    BlockPullPlan,
    BlockRange,
    TimestampRange,
)
from spice.config import AcquireConfig, WorkflowTask, coerce_features_config
from spice.config.groups import load_named_group_payload
from spice.core.reporting import Reporter
from spice.storage.catalog.index import list_dataset_records
from spice.storage.corpus import list_acquire_runs, load_dataset_manifest
from spice.storage.selectors import DatasetSelector
from spice.workflows.acquire import run as run_acquire
from spice.workflows.preparation import resolve_acquire_producer_roots
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
    roots = resolve_acquire_producer_roots(config)
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
        def __init__(self, rpc_endpoint, chain, *, include_priority_fees=False) -> None:
            del rpc_endpoint
            assert include_priority_fees is False
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

    summary = load_dataset_manifest(roots.corpus.state_db_path)
    runs = list_acquire_runs(roots.corpus.state_db_path)
    assert roots.corpus.state_db_path.is_file()
    assert summary.splits.evaluation.validation.rows == evaluation_plan.expected_rows
    assert summary.dataset.id == roots.corpus.dataset_id
    assert summary.dataset.name == config.dataset.name
    assert summary.chain.name == config.chain.name
    assert len(runs) == 1
    assert runs[0].facts.resolved_capability_samples >= config.problem.sample_count
    assert roots.corpus.history_dir.is_dir()
    assert roots.corpus.evaluation_dir.is_dir()
    datasets = list_dataset_records(
        roots.corpus.storage_root,
        selector=DatasetSelector(
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
        ),
    )
    assert len(datasets) == 1
    assert datasets[0].dataset_id == roots.corpus.dataset_id


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
    roots = resolve_acquire_producer_roots(config)
    stage_root = roots.corpus.root_path.parent / f".{roots.corpus.dataset_id}.acquire-staging"
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
            start=config.history_window_end_timestamp - 400 * 12,
            end=config.history_window_end_timestamp,
        ),
        start_block=100,
        expected_rows=400,
        chunk_size=config.acquisition.chunk_size,
    )
    requested_ranges: list[tuple[int, int]] = []
    fail_history_tail = True

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain, *, include_priority_fees=False) -> None:
            del rpc_endpoint
            assert include_priority_fees is False
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
    with pytest.raises(RuntimeError, match="exceeded 8 transient retry attempts"):
        run_acquire(config)

    assert stage_root.is_dir()
    assert len(list(stage_root.rglob("*.parquet"))) == 2
    assert roots.corpus.state_db_path.exists() is False
    assert requested_ranges.count((100, 104)) == 1
    assert requested_ranges.count((104, 108)) == 1

    fail_history_tail = False
    run_acquire(config)

    assert stage_root.exists() is False
    assert roots.corpus.state_db_path.is_file()
    assert roots.corpus.history_dir.is_dir()
    assert roots.corpus.evaluation_dir.is_dir()
    assert requested_ranges.count((100, 104)) == 1
    assert requested_ranges.count((104, 108)) == 1
    assert (108, 112) in requested_ranges


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
    roots = resolve_acquire_producer_roots(config)
    output = StringIO()
    reporter = Reporter(stream=output)

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain, *, include_priority_fees=False) -> None:
            del rpc_endpoint, chain, include_priority_fees

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
    assert not roots.corpus.state_db_path.exists()


def test_acquire_maps_priority_fee_source_requirement_to_rpc_enrichment(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    base_config = _load_test_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    config = base_config.model_copy(
        update={
            "features": coerce_features_config(
                load_named_group_payload("core_fee_dynamics_with_priority_fee", "features")
            ),
            "acquisition": base_config.acquisition.model_copy(update={"dry_run": True}),
        }
    )
    include_flags: list[bool] = []

    class FakeAcquireClient:
        def __init__(self, rpc_endpoint, chain, *, include_priority_fees=False) -> None:
            del rpc_endpoint, chain
            include_flags.append(include_priority_fees)

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

    run_acquire(config)

    assert include_flags == [True]

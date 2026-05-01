from __future__ import annotations

import asyncio
import math
from typing import cast

import polars as pl
import pytest

from spice.acquisition import AcquisitionPullController, BlockPullPlan, BlockRange, TimestampRange
from spice.config import AcquireConfig, WorkflowTask
from spice.corpus.assembly import CorpusAssemblyRequest, assemble_corpus
from spice.corpus.contract import canonicalize_block_frame
from spice.corpus.io import load_block_frame
from spice.corpus.split_materialization import (
    CorpusSplitMaterializationSpec,
    CorpusSplitOutcome,
    ensure_evaluation_split,
    ensure_history_split,
    pull_plan_to_frame,
    write_block_dataset_dir,
)
from spice.storage.root_producer_handles import resolve_acquire_producer_roots
from tests.dataset_helpers import make_block_rows


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


def _plan_for_window(
    window: TimestampRange,
    *,
    start_block: int,
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
            block_range=block_range,
            expected_rows=block_range.count,
        )

    async def get_block_rows(self, start: int, end: int):
        del start, end
        raise AssertionError("dry run must not fetch rows")


def test_assemble_corpus_dry_run_returns_plan_without_writes(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )
    config.acquisition.dry_run = True
    roots = resolve_acquire_producer_roots(config)
    source = _PlanningSource(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        )
    )

    result = asyncio.run(
        assemble_corpus(
            CorpusAssemblyRequest(config=config, roots=roots),
            source,
        )
    )

    assert result.mode == "dry_run"
    assert result.history_plan.expected_rows > 0
    assert result.evaluation_plan.expected_rows > 0
    assert result.manifest is None
    assert roots.corpus.state_db_path.exists() is False


def test_assemble_corpus_preserves_staging_on_failure(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    override = acquire_override()
    override["acquisition"] = {
        "chunk_size": 4,
        "rpc": {
            "batch_size": 4,
            "concurrency": 1,
            "min_batch_size": 4,
            "concurrency_rungs": [1],
        },
    }
    config = _load_acquire_config(load_workflow_config, tmp_path, override=override)
    roots = resolve_acquire_producer_roots(config)
    source = _PlanningSource(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        )
    )
    messages: list[str] = []

    with pytest.raises(AssertionError, match="dry run must not fetch rows"):
        asyncio.run(
            assemble_corpus(
                CorpusAssemblyRequest(config=config, roots=roots),
                source,
                status=messages.append,
            )
        )

    assert (
        roots.corpus.root_path.parent / f".{roots.corpus.dataset_id}.acquire-staging"
    ).is_dir()
    assert "history downloading" in messages


async def _exercise_history_split_materialization(
    config: AcquireConfig,
    tmp_path,
) -> tuple[CorpusSplitOutcome, list[str]]:
    plan = _plan_for_window(
        TimestampRange(
            start=config.history_window_end_timestamp - 120,
            end=config.history_window_end_timestamp,
        ),
        start_block=100,
        expected_rows=10,
    )

    class Source:
        async def close(self) -> None:
            return None

        async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
            del sample_size
            return 12.0

        async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
            del window
            return plan

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
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=plan.window.start + (start - plan.block_range.start) * 12,
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    result = await ensure_history_split(
        materialization=CorpusSplitMaterializationSpec(
            chain_name=config.chain.name,
            expected_chain_id=config.chain.runtime.chain_id,
            chunk_size=4,
        ),
        block_source=Source(),
        output_dir=tmp_path / "history",
        working_dir=tmp_path / "work",
        history_plan=plan,
        controller=AcquisitionPullController.from_config(config.acquisition),
    )
    return result.outcome, sorted(path.name for path in result.path.glob("*.parquet"))


def test_history_split_materialization_uses_split_spec(
    tmp_path,
    load_workflow_config,
    acquire_override,
) -> None:
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=acquire_override(),
    )

    outcome, filenames = asyncio.run(_exercise_history_split_materialization(config, tmp_path))

    assert outcome is CorpusSplitOutcome.CREATED
    assert filenames
    assert all(name.startswith(f"{config.chain.name}__blocks__") for name in filenames)


def test_split_materialization_parquet_sink_resumes_completed_prefix(
    tmp_path,
) -> None:
    output_dir = tmp_path / "pull"
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=2_000),
        start_block=100,
        expected_rows=4,
    )
    write_block_dataset_dir(
        output_dir,
        frame=canonicalize_block_frame(
            pl.DataFrame(
                make_block_rows(2, start_block=100, start_timestamp=1_000, chain_id=1)
            )
        ),
        chunk_size=2,
        chain_name="ethereum",
    )
    requests: list[tuple[int, int]] = []

    class Source:
        async def get_block_rows(self, start: int, end: int):
            requests.append((start, end))
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=1_000 + (start - 100) * 12,
                chain_id=1,
            )

    frame = asyncio.run(
        pull_plan_to_frame(
            block_source=Source(),
            plan=plan,
            output_dir=output_dir,
            materialization=CorpusSplitMaterializationSpec(
                chain_name="ethereum",
                expected_chain_id=1,
                chunk_size=2,
            ),
            controller=AcquisitionPullController(
                configured_batch_size=2,
                min_batch_size=1,
                concurrency_rungs=(1,),
                configured_concurrency=1,
            ),
        )
    )

    assert requests == [(102, 104)]
    assert frame["block_number"].to_list() == [100, 101, 102, 103]
    assert load_block_frame(output_dir)["block_number"].to_list() == [100, 101, 102, 103]


def test_evaluation_split_extension_reuses_whole_existing_chunks(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    working_dir = tmp_path / "work"
    materialization = CorpusSplitMaterializationSpec(
        chain_name="ethereum",
        expected_chain_id=1,
        chunk_size=2,
    )
    write_block_dataset_dir(
        output_dir,
        frame=canonicalize_block_frame(
            pl.DataFrame(
                make_block_rows(4, start_block=100, start_timestamp=1_000, chain_id=1)
            )
        ),
        chunk_size=2,
        chain_name="ethereum",
    )
    plan = BlockPullPlan(
        window=TimestampRange(start=1_012, end=1_060),
        block_range=BlockRange(start=101, end=105),
        expected_rows=4,
    )
    requests: list[tuple[int, int]] = []

    class Source:
        async def get_block_rows(self, start: int, end: int):
            requests.append((start, end))
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=1_000 + (start - 100) * 12,
                chain_id=1,
            )

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

    result = asyncio.run(
        ensure_evaluation_split(
            materialization=materialization,
            block_source=Source(),
            output_dir=output_dir,
            working_dir=working_dir,
            evaluation_plan=plan,
            controller=AcquisitionPullController(
                configured_batch_size=2,
                min_batch_size=1,
                concurrency_rungs=(1,),
                configured_concurrency=1,
            ),
        )
    )

    assert result.outcome is CorpusSplitOutcome.EXTENDED
    assert requests == [(104, 105)]
    assert sorted(path.name for path in result.path.glob("*.parquet")) == [
        "ethereum__blocks__101_to_101.parquet",
        "ethereum__blocks__102_to_103.parquet",
        "ethereum__blocks__104_to_104.parquet",
    ]
    assert load_block_frame(result.path)["block_number"].to_list() == [101, 102, 103, 104]


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
    config = _load_acquire_config(
        load_workflow_config,
        tmp_path,
        override=override,
    )
    roots = resolve_acquire_producer_roots(config)
    evaluation_plan = _plan_for_window(
        TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        start_block=10_000,
        expected_rows=32,
    )
    history_plans = [
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 50 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=1_000,
            expected_rows=50,
        ),
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 100 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=950,
            expected_rows=100,
        ),
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 150 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=900,
            expected_rows=150,
        ),
        _plan_for_window(
            TimestampRange(
                start=config.history_window_end_timestamp - 200 * 12,
                end=config.history_window_end_timestamp,
            ),
            start_block=850,
            expected_rows=200,
        ),
    ]
    partial_ranges: list[tuple[int, int]] = []
    requested_ranges: list[tuple[int, int]] = []
    history_windows: list[TimestampRange] = []
    if final_sample_count is None:
        sample_counts = [1, config.problem.sample_count]
    else:
        sample_counts = [1, final_sample_count, final_sample_count, final_sample_count]
    resolved_capability_samples = iter(sample_counts)
    history_plan_calls = 0

    class Source:
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
            requested_ranges.append((start, end))
            if start >= evaluation_plan.block_range.start:
                plan = evaluation_plan
            else:
                plan = next(
                    plan
                    for plan in reversed(history_plans)
                    if start >= plan.block_range.start
                )
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=plan.window.start + (start - plan.block_range.start) * 12,
                chain_id=config.chain.runtime.chain_id,
                block_interval_seconds=12,
            )

    monkeypatch.setattr(
        "spice.corpus.planning.CorpusCapabilityPlanningContext.count_valid_history_samples",
        lambda self, history_dir: next(resolved_capability_samples),
    )

    request = CorpusAssemblyRequest(config=config, roots=roots)
    if expect_error:
        with pytest.raises(RuntimeError, match="under-requested capability samples"):
            asyncio.run(assemble_corpus(request, Source()))
    else:
        asyncio.run(assemble_corpus(request, Source()))
    return partial_ranges, requested_ranges, history_windows, history_plan_calls


def test_assemble_corpus_refills_missing_history_prefix(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    acquire_override,
) -> None:
    partial_ranges, requested_ranges, _, _ = _exercise_short_history_refill(
        tmp_path,
        monkeypatch,
        load_workflow_config,
        acquire_override,
    )

    assert partial_ranges == [(950, 1_000)]
    assert (1_000, 1_050) in requested_ranges
    assert (950, 1_000) in requested_ranges
    assert (950, 1_050) not in requested_ranges


def test_assemble_corpus_fails_after_bounded_short_refills(
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
    assert history_plan_calls == 4

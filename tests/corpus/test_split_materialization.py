from __future__ import annotations

import asyncio
import math
from typing import cast

import polars as pl
import pytest

from spice.acquisition import AcquisitionPullController, BlockPullPlan, BlockRange, TimestampRange
from spice.corpus.contract import CanonicalBlockRow, canonicalize_block_frame
from spice.corpus.io import load_block_frame, write_block_file
from spice.corpus.split_materialization import (
    CorpusSplitIntent,
    CorpusSplitKind,
    CorpusSplitMaterializationSession,
    CorpusSplitMaterializationSpec,
    CorpusSplitOutcome,
)
from tests.dataset_helpers import make_block_rows


def _plan_for_window(
    window: TimestampRange,
    *,
    start_block: int,
    block_interval_seconds: int = 12,
    row_count: int | None = None,
) -> BlockPullPlan:
    resolved_row_count = (
        row_count
        if row_count is not None
        else max(1, math.ceil((window.end - window.start) / block_interval_seconds))
    )
    return BlockPullPlan(
        window=window,
        block_range=BlockRange(start=start_block, end=start_block + resolved_row_count),
    )


def _write_block_dataset_dir(
    output_dir,
    *,
    start_block: int,
    row_count: int,
    start_timestamp: int,
    chain_id: int = 1,
    chunk_size: int = 2,
    chain_name: str = "ethereum",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = canonicalize_block_frame(
        pl.DataFrame(
            make_block_rows(
                row_count,
                start_block=start_block,
                start_timestamp=start_timestamp,
                chain_id=chain_id,
            )
        )
    )
    for start in range(0, frame.height, chunk_size):
        chunk = frame.slice(start, min(chunk_size, frame.height - start))
        first = int(chunk["block_number"][0])
        last = int(chunk["block_number"][-1])
        write_block_file(output_dir / f"{chain_name}__blocks__{first}_to_{last}.parquet", chunk)


def _controller() -> AcquisitionPullController:
    return AcquisitionPullController(
        configured_batch_size=2,
        min_batch_size=1,
        concurrency_rungs=(1,),
        configured_concurrency=1,
    )


def _materialization() -> CorpusSplitMaterializationSpec:
    return CorpusSplitMaterializationSpec(
        chain_name="ethereum",
        expected_chain_id=1,
        chunk_size=2,
    )


class _RangeSource:
    def __init__(self, *, timestamp_base: int = 1_000) -> None:
        self.timestamp_base = timestamp_base
        self.requests: list[tuple[int, int]] = []
        self.partial_ranges: list[tuple[int, int]] = []

    async def get_block_rows(self, start: int, end: int):
        self.requests.append((start, end))
        return cast(
            list[CanonicalBlockRow],
            make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=self.timestamp_base + (start - 100) * 12,
                chain_id=1,
            ),
        )

    def plan_block_range(
        self,
        block_range: BlockRange,
        *,
        window: TimestampRange,
    ) -> BlockPullPlan:
        self.partial_ranges.append((block_range.start, block_range.end))
        return BlockPullPlan(
            window=window,
            block_range=block_range
        )

    async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
        del sample_size
        return 12.0

    async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
        return _plan_for_window(window, start_block=100)

    async def close(self) -> None:
        return None


def _session(source: _RangeSource) -> CorpusSplitMaterializationSession:
    return CorpusSplitMaterializationSession(
        materialization=_materialization(),
        block_source=source,
        controller=_controller(),
    )


def test_history_split_materialization_creates_dataset(tmp_path) -> None:
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_120),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.HISTORY,
                output_dir=tmp_path / "history",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.CREATED
    assert result.promote_dir == result.path
    assert sorted(path.name for path in result.path.glob("*.parquet"))


def test_history_split_materialization_reuses_matching_staged_dataset(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "work" / "history",
        start_block=100,
        row_count=4,
        start_timestamp=1_000,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.HISTORY,
                output_dir=tmp_path / "history",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.CREATED
    assert result.promote_dir == tmp_path / "work" / "history"
    assert source.requests == []


def test_history_split_materialization_ignores_stale_clean_staged_dataset(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "work" / "history",
        start_block=100,
        row_count=2,
        start_timestamp=1_000,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.HISTORY,
                output_dir=tmp_path / "history",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.CREATED
    assert source.requests == [(100, 102), (102, 104)]
    assert load_block_frame(result.path)["block_number"].to_list() == [100, 101, 102, 103]


def test_history_split_materialization_rejects_invalid_staged_dataset(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "work" / "history",
        start_block=100,
        row_count=2,
        start_timestamp=1_000,
        chain_id=999,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    with pytest.raises(RuntimeError, match="Cannot resume invalid staged history dataset"):
        asyncio.run(
            _session(source).fulfill(
                CorpusSplitIntent(
                    kind=CorpusSplitKind.HISTORY,
                    output_dir=tmp_path / "history",
                    working_dir=tmp_path / "work",
                    plan=plan,
                )
            )
        )

    assert source.requests == []
    assert not (tmp_path / "history").exists()


def test_history_split_materialization_reuses_committed_superset(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "history",
        start_block=90,
        row_count=10,
        start_timestamp=880,
    )
    source = _RangeSource()
    plan = BlockPullPlan(
        window=TimestampRange(start=1_000, end=1_120),
        block_range=BlockRange(start=95, end=100)
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.HISTORY,
                output_dir=tmp_path / "history",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.REUSED
    assert result.path == tmp_path / "history"
    assert result.promote_dir is None
    assert source.requests == []


def test_history_split_materialization_extends_committed_prefix(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "history",
        start_block=102,
        row_count=2,
        start_timestamp=1_024,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.HISTORY,
                output_dir=tmp_path / "history",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.EXTENDED
    assert source.partial_ranges == [(100, 102)]
    assert source.requests == [(100, 102)]
    assert load_block_frame(result.path)["block_number"].to_list() == [100, 101, 102, 103]


def test_evaluation_split_materialization_reuses_exact_committed_dataset(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "evaluation",
        start_block=100,
        row_count=4,
        start_timestamp=1_000,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.EVALUATION,
                output_dir=tmp_path / "evaluation",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.REUSED
    assert result.promote_dir is None
    assert source.requests == []


def test_evaluation_split_materialization_extends_overlap(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "evaluation",
        start_block=100,
        row_count=4,
        start_timestamp=1_000,
    )
    source = _RangeSource()
    plan = BlockPullPlan(
        window=TimestampRange(start=1_012, end=1_060),
        block_range=BlockRange(start=101, end=105)
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.EVALUATION,
                output_dir=tmp_path / "evaluation",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.EXTENDED
    assert source.partial_ranges == [(104, 105)]
    assert source.requests == [(104, 105)]
    assert sorted(path.name for path in result.path.glob("*.parquet")) == [
        "ethereum__blocks__101_to_101.parquet",
        "ethereum__blocks__102_to_103.parquet",
        "ethereum__blocks__104_to_104.parquet",
    ]
    assert load_block_frame(result.path)["block_number"].to_list() == [101, 102, 103, 104]


def test_evaluation_split_materialization_extends_prefix_and_suffix(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "evaluation",
        start_block=100,
        row_count=4,
        start_timestamp=1_000,
    )
    source = _RangeSource()
    plan = BlockPullPlan(
        window=TimestampRange(start=976, end=1_072),
        block_range=BlockRange(start=98, end=106)
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.EVALUATION,
                output_dir=tmp_path / "evaluation",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.EXTENDED
    assert source.partial_ranges == [(98, 100), (104, 106)]
    assert source.requests == [(98, 100), (104, 106)]
    assert sorted(path.name for path in result.path.glob("*.parquet")) == [
        "ethereum__blocks__100_to_101.parquet",
        "ethereum__blocks__102_to_103.parquet",
        "ethereum__blocks__104_to_105.parquet",
        "ethereum__blocks__98_to_99.parquet",
    ]
    assert load_block_frame(result.path)["block_number"].to_list() == [
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
    ]


def test_evaluation_split_materialization_rebuilds_overlap_outside_window(
    tmp_path,
) -> None:
    _write_block_dataset_dir(
        tmp_path / "evaluation",
        start_block=100,
        row_count=4,
        start_timestamp=0,
    )
    source = _RangeSource()
    plan = BlockPullPlan(
        window=TimestampRange(start=1_012, end=1_060),
        block_range=BlockRange(start=101, end=105)
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.EVALUATION,
                output_dir=tmp_path / "evaluation",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.REBUILT
    assert source.requests == [(101, 103), (103, 105)]
    assert load_block_frame(result.path)["timestamp"].to_list() == [
        1_012,
        1_024,
        1_036,
        1_048,
    ]


def test_evaluation_split_materialization_rebuilds_stale_exact_range(tmp_path) -> None:
    _write_block_dataset_dir(
        tmp_path / "evaluation",
        start_block=100,
        row_count=4,
        start_timestamp=0,
    )
    source = _RangeSource()
    plan = _plan_for_window(
        TimestampRange(start=1_000, end=1_048),
        start_block=100
    )

    result = asyncio.run(
        _session(source).fulfill(
            CorpusSplitIntent(
                kind=CorpusSplitKind.EVALUATION,
                output_dir=tmp_path / "evaluation",
                working_dir=tmp_path / "work",
                plan=plan,
            )
        )
    )

    assert result.outcome is CorpusSplitOutcome.REBUILT
    assert source.requests == [(100, 102), (102, 104)]
    assert load_block_frame(result.path)["timestamp"].to_list() == [1_000, 1_012, 1_024, 1_036]

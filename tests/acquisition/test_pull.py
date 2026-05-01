from __future__ import annotations

import asyncio

import pytest

from spice.acquisition.errors import (
    OversizedAcquisitionRequestError,
    TransientAcquisitionError,
)
from spice.acquisition.pull import AcquisitionPullController, pull_block_range
from spice.acquisition.types import BlockPullPlan, BlockRange, TimestampRange
from tests.dataset_helpers import make_block_rows


def _plan(start: int, end: int) -> BlockPullPlan:
    return BlockPullPlan(
        window=TimestampRange(start=1_000, end=2_000),
        block_range=BlockRange(start=start, end=end),
        expected_rows=end - start,
    )


class _FakeBlockSource:
    def __init__(self, *, chain_id: int = 1) -> None:
        self.chain_id = chain_id
        self.requests: list[tuple[int, int]] = []

    async def get_block_rows(self, start: int, end: int):
        self.requests.append((start, end))
        return make_block_rows(
            end - start,
            start_block=start,
            start_timestamp=1_000 + (start - 100) * 12,
            chain_id=self.chain_id,
        )


class _FakeSink:
    def __init__(self, *, completed_until: int | None = None) -> None:
        self.completed_until = completed_until
        self.rows: list[dict[str, object]] = []
        self.finished = False

    def completed_prefix_end(self, plan: BlockPullPlan) -> int:
        return plan.block_range.start if self.completed_until is None else self.completed_until

    def write_rows(self, rows) -> None:
        self.rows.extend(rows)

    def finish(self) -> None:
        self.finished = True


def test_pull_writes_ordered_chunks_with_out_of_order_completion(tmp_path) -> None:
    del tmp_path

    class DelayedSource(_FakeBlockSource):
        async def get_block_rows(self, start: int, end: int):
            await asyncio.sleep(0.02 if start == 100 else 0)
            return await super().get_block_rows(start, end)

    source = DelayedSource()
    sink = _FakeSink()
    asyncio.run(
        pull_block_range(
            source,
            plan=_plan(100, 104),
            controller=AcquisitionPullController(
                configured_batch_size=2,
                min_batch_size=1,
                concurrency_rungs=(1, 2),
                configured_concurrency=2,
            ),
            sink=sink,
        )
    )

    assert [int(row["block_number"]) for row in sink.rows] == [100, 101, 102, 103]
    assert sink.finished is True


def test_pull_resumes_completed_prefix(tmp_path) -> None:
    del tmp_path

    source = _FakeBlockSource()
    sink = _FakeSink(completed_until=102)

    asyncio.run(
        pull_block_range(
            source,
            plan=_plan(100, 104),
            controller=AcquisitionPullController(
                configured_batch_size=2,
                min_batch_size=1,
                concurrency_rungs=(1,),
                configured_concurrency=1,
            ),
            sink=sink,
        )
    )

    assert source.requests == [(102, 104)]
    assert [int(row["block_number"]) for row in sink.rows] == [102, 103]


def test_pull_rejects_rows_that_do_not_match_requested_block_range(tmp_path) -> None:
    del tmp_path

    class ShiftedSource(_FakeBlockSource):
        async def get_block_rows(self, start: int, end: int):
            self.requests.append((start, end))
            return make_block_rows(
                end - start,
                start_block=start + 1,
                start_timestamp=1_000 + (start - 100) * 12,
                chain_id=self.chain_id,
            )

    sink = _FakeSink()
    with pytest.raises(RuntimeError, match="do not match requested range"):
        asyncio.run(
            pull_block_range(
                ShiftedSource(),
                plan=_plan(100, 102),
                controller=AcquisitionPullController(
                    configured_batch_size=2,
                    min_batch_size=1,
                    concurrency_rungs=(1,),
                    configured_concurrency=1,
                ),
                sink=sink,
            )
        )

    assert sink.rows == []
    assert sink.finished is False


def test_pull_splits_oversized_requests_and_backs_off(tmp_path) -> None:
    del tmp_path

    class OversizeOnceSource(_FakeBlockSource):
        async def get_block_rows(self, start: int, end: int):
            self.requests.append((start, end))
            if (start, end) == (100, 104):
                raise OversizedAcquisitionRequestError("batch too large")
            return make_block_rows(
                end - start,
                start_block=start,
                start_timestamp=1_000 + (start - 100) * 12,
                chain_id=1,
            )

    source = OversizeOnceSource()
    sink = _FakeSink()
    controller = AcquisitionPullController(
        configured_batch_size=4,
        min_batch_size=2,
        concurrency_rungs=(1,),
        configured_concurrency=1,
    )

    asyncio.run(
        pull_block_range(
            source,
            plan=_plan(100, 104),
            controller=controller,
            sink=sink,
        )
    )

    assert source.requests == [(100, 104), (100, 102), (102, 104)]
    assert controller.current_batch_size == 2
    assert controller.oversize_backoffs == 1


def test_pull_fails_after_transient_retry_limit(tmp_path) -> None:
    del tmp_path

    class AlwaysTransientSource(_FakeBlockSource):
        async def get_block_rows(self, start: int, end: int):
            self.requests.append((start, end))
            raise TransientAcquisitionError("timeout")

    source = AlwaysTransientSource()
    with pytest.raises(RuntimeError, match="exceeded 8 transient retry attempts"):
        asyncio.run(
            pull_block_range(
                source,
                plan=_plan(100, 101),
                controller=AcquisitionPullController(
                    configured_batch_size=1,
                    min_batch_size=1,
                    concurrency_rungs=(1,),
                    configured_concurrency=1,
                ),
                sink=_FakeSink(),
            )
        )
    assert len(source.requests) == 8


def test_pull_cancellation_cancels_in_flight_tasks(tmp_path) -> None:
    del tmp_path

    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingSource(_FakeBlockSource):
        async def get_block_rows(self, start: int, end: int):
            self.requests.append((start, end))
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def exercise() -> None:
        source = HangingSource()
        task = asyncio.create_task(
            pull_block_range(
                source,
                plan=_plan(100, 101),
                controller=AcquisitionPullController(
                    configured_batch_size=1,
                    min_batch_size=1,
                    concurrency_rungs=(1,),
                    configured_concurrency=1,
                ),
                sink=_FakeSink(),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    asyncio.run(exercise())


def test_pull_controller_backs_off_and_recovers_concurrency() -> None:
    controller = AcquisitionPullController(
        configured_batch_size=8,
        min_batch_size=2,
        concurrency_rungs=(1, 2, 4),
        configured_concurrency=4,
    )

    assert controller.record_oversize_failure() == 4
    assert controller.current_batch_size == 4
    assert controller.record_oversize_failure() == 2
    assert controller.record_oversize_failure() is None

    assert controller.record_transient_failure() is None
    assert controller.record_transient_failure() == 2
    for _ in range(64):
        recovered = controller.record_success()
    assert recovered == 4

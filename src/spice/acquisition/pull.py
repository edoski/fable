"""Generic block pulling scheduler and retry logic."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Protocol

from ..config.models import AcquisitionConfig
from ..corpus.contract import CanonicalBlockRow
from .errors import OversizedAcquisitionRequestError, TransientAcquisitionError
from .types import AcquisitionRuntimeSnapshot, BlockPullPlan, BlockRowSource

MAX_ACQUISITION_ATTEMPTS_PER_RANGE = 8
TRANSIENT_FAILURE_WINDOW = 32
TRANSIENT_FAILURE_WINDOW_THRESHOLD = 3
TRANSIENT_FAILURE_STREAK_THRESHOLD = 2
SUCCESS_STREAK_FOR_RECOVERY = 64


@dataclass(slots=True)
class AcquisitionPullController:
    configured_batch_size: int
    min_batch_size: int
    concurrency_rungs: tuple[int, ...]
    configured_concurrency: int
    current_batch_size: int = field(init=False)
    _configured_concurrency_index: int = field(init=False, repr=False)
    _current_concurrency_index: int = field(init=False, repr=False)
    _success_streak: int = field(default=0, init=False, repr=False)
    _transient_streak: int = field(default=0, init=False, repr=False)
    _recent_transient_attempts: deque[int] = field(init=False, repr=False)
    oversize_error_count: int = 0
    transient_error_count: int = 0
    oversize_backoffs: int = 0
    transient_backoffs: int = 0
    concurrency_recoveries: int = 0

    def __post_init__(self) -> None:
        if self.min_batch_size > self.configured_batch_size:
            raise ValueError("acquisition.rpc.min_batch_size must be <= batch_size")
        if not self.concurrency_rungs:
            raise ValueError("acquisition.rpc.concurrency_rungs must not be empty")
        if tuple(sorted(self.concurrency_rungs)) != self.concurrency_rungs:
            raise ValueError("acquisition.rpc.concurrency_rungs must be sorted ascending")
        if len(set(self.concurrency_rungs)) != len(self.concurrency_rungs):
            raise ValueError("acquisition.rpc.concurrency_rungs must not contain duplicates")
        if self.configured_concurrency not in self.concurrency_rungs:
            raise ValueError("acquisition.rpc.concurrency must be present in concurrency_rungs")

        self.current_batch_size = self.configured_batch_size
        self._configured_concurrency_index = self.concurrency_rungs.index(
            self.configured_concurrency
        )
        self._current_concurrency_index = self._configured_concurrency_index
        self._recent_transient_attempts = deque(maxlen=TRANSIENT_FAILURE_WINDOW)

    @classmethod
    def from_config(cls, config: AcquisitionConfig) -> AcquisitionPullController:
        return cls(
            configured_batch_size=config.rpc.batch_size,
            min_batch_size=config.rpc.min_batch_size,
            concurrency_rungs=tuple(config.rpc.concurrency_rungs),
            configured_concurrency=config.rpc.concurrency,
        )

    @property
    def current_concurrency(self) -> int:
        return self.concurrency_rungs[self._current_concurrency_index]

    def record_success(self) -> int | None:
        self._success_streak += 1
        self._transient_streak = 0
        self._recent_transient_attempts.append(0)
        if (
            self._success_streak < SUCCESS_STREAK_FOR_RECOVERY
            or self._current_concurrency_index >= self._configured_concurrency_index
        ):
            return None

        self._current_concurrency_index += 1
        self._success_streak = 0
        self._recent_transient_attempts.clear()
        self.concurrency_recoveries += 1
        return self.current_concurrency

    def record_oversize_failure(self) -> int | None:
        self.oversize_error_count += 1
        self._success_streak = 0
        self._transient_streak = 0

        next_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        if next_batch_size >= self.current_batch_size:
            return None

        self.current_batch_size = next_batch_size
        self.oversize_backoffs += 1
        return self.current_batch_size

    def record_transient_failure(self) -> int | None:
        self.transient_error_count += 1
        self._success_streak = 0
        self._transient_streak += 1
        self._recent_transient_attempts.append(1)
        if self._current_concurrency_index == 0:
            return None

        if (
            self._transient_streak < TRANSIENT_FAILURE_STREAK_THRESHOLD
            and sum(self._recent_transient_attempts) < TRANSIENT_FAILURE_WINDOW_THRESHOLD
        ):
            return None

        self._current_concurrency_index -= 1
        self._transient_streak = 0
        self._recent_transient_attempts.clear()
        self.transient_backoffs += 1
        return self.current_concurrency

    def snapshot(self) -> AcquisitionRuntimeSnapshot:
        return AcquisitionRuntimeSnapshot(
            configured_batch_size=self.configured_batch_size,
            final_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            configured_concurrency=self.configured_concurrency,
            final_concurrency=self.current_concurrency,
            concurrency_rungs=self.concurrency_rungs,
            oversize_error_count=self.oversize_error_count,
            transient_error_count=self.transient_error_count,
            oversize_backoffs=self.oversize_backoffs,
            transient_backoffs=self.transient_backoffs,
            concurrency_recoveries=self.concurrency_recoveries,
        )


@dataclass(frozen=True, slots=True, order=True)
class _BatchRequest:
    start: int
    end: int
    attempts: int = 0

    @property
    def size(self) -> int:
        return self.end - self.start

    def retry(self) -> _BatchRequest:
        return _BatchRequest(
            start=self.start,
            end=self.end,
            attempts=self.attempts + 1,
        )


@dataclass(frozen=True, slots=True)
class _CompletedBatch:
    start: int
    end: int
    rows: list[CanonicalBlockRow]


class BlockPullSink(Protocol):
    def completed_prefix_end(self, plan: BlockPullPlan) -> int: ...

    def write_rows(self, rows: list[CanonicalBlockRow]) -> None: ...

    def finish(self) -> None: ...


async def pull_block_range(
    block_source: BlockRowSource,
    *,
    plan: BlockPullPlan,
    controller: AcquisitionPullController,
    sink: BlockPullSink,
) -> BlockPullPlan:
    pending_requests: list[_BatchRequest] = []
    in_flight: dict[asyncio.Task[list[CanonicalBlockRow]], _BatchRequest] = {}
    completed_results: dict[int, _CompletedBatch] = {}
    resumed_until = sink.completed_prefix_end(plan)
    next_request_start = resumed_until
    next_write_start = resumed_until

    try:
        while next_write_start < plan.block_range.end:
            while len(in_flight) < controller.current_concurrency:
                request = _next_request(
                    pending_requests,
                    next_request_start=next_request_start,
                    range_end=plan.block_range.end,
                    batch_size=controller.current_batch_size,
                )
                if request is None:
                    break
                if request.start >= next_request_start:
                    next_request_start = request.end
                task = asyncio.create_task(block_source.get_block_rows(request.start, request.end))
                in_flight[task] = request

            if not in_flight:
                raise RuntimeError("Acquisition scheduler stalled before completing block range")

            done, _ = await asyncio.wait(
                in_flight,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                request = in_flight.pop(task)
                try:
                    rows = task.result()
                except Exception as exc:
                    if isinstance(exc, OversizedAcquisitionRequestError):
                        next_batch_size = controller.record_oversize_failure()
                        if next_batch_size is None or request.size <= controller.min_batch_size:
                            raise RuntimeError(
                                "Acquisition batch remained too large at the configured minimum "
                                f"batch size ({controller.min_batch_size})"
                            ) from exc
                        for retry_request in _split_request(request, batch_size=next_batch_size):
                            heappush(pending_requests, retry_request)
                        continue

                    if _is_transient_error(exc):
                        controller.record_transient_failure()
                        if request.attempts + 1 >= MAX_ACQUISITION_ATTEMPTS_PER_RANGE:
                            raise RuntimeError(
                                f"Acquisition range {request.start}..{request.end} exceeded "
                                f"{MAX_ACQUISITION_ATTEMPTS_PER_RANGE} transient retry attempts"
                            ) from exc
                        heappush(pending_requests, request.retry())
                        continue

                    raise

                _validate_pulled_rows(request, rows)
                controller.record_success()

                completed_results[request.start] = _CompletedBatch(
                    start=request.start,
                    end=request.end,
                    rows=rows,
                )
                while next_write_start in completed_results:
                    finished_batch = completed_results.pop(next_write_start)
                    sink.write_rows(finished_batch.rows)
                    next_write_start = finished_batch.end
        sink.finish()

        return plan
    finally:
        for task in in_flight:
            task.cancel()
        current_task = asyncio.current_task()
        if in_flight and (current_task is None or current_task.cancelling() == 0):
            await asyncio.gather(*in_flight, return_exceptions=True)


def _next_request(
    pending_requests: list[_BatchRequest],
    *,
    next_request_start: int,
    range_end: int,
    batch_size: int,
) -> _BatchRequest | None:
    if pending_requests:
        return heappop(pending_requests)
    if next_request_start >= range_end:
        return None
    return _BatchRequest(
        start=next_request_start,
        end=min(next_request_start + batch_size, range_end),
    )


def _validate_pulled_rows(
    request: _BatchRequest,
    rows: list[CanonicalBlockRow],
) -> None:
    if len(rows) != request.size:
        raise RuntimeError(
            f"Expected {request.size} rows for {request.start}..{request.end}, got {len(rows)}"
        )
    block_numbers = [int(row["block_number"]) for row in rows]
    expected = list(range(request.start, request.end))
    if block_numbers != expected:
        raise RuntimeError(
            "Pulled block rows do not match requested range "
            f"{request.start}..{request.end}: got {block_numbers}"
        )


def _split_request(
    request: _BatchRequest,
    *,
    batch_size: int,
) -> list[_BatchRequest]:
    return [
        _BatchRequest(
            start=batch_start,
            end=min(batch_start + batch_size, request.end),
            attempts=request.attempts + 1,
        )
        for batch_start in range(request.start, request.end, batch_size)
    ]


def _is_transient_error(exc: BaseException) -> bool:
    return isinstance(exc, (TransientAcquisitionError, asyncio.TimeoutError, TimeoutError, OSError))

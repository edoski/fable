"""Async web3.py-backed helpers for canonical block acquisition."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil
from pathlib import Path
from typing import Literal, SupportsInt, cast

import aiohttp
import polars as pl
from web3 import AsyncWeb3
from web3.exceptions import Web3RPCError

from ..config import AcquisitionConfig, ChainSpec, ProviderSpec
from ..core.console import NullReporter, Reporter
from ..data.block_contract import (
    CanonicalBlockRow,
    RpcBlock,
    build_canonical_block_row,
    canonicalize_block_frame,
)
from ..data.io import write_block_file
from .provider import build_async_web3

MAX_RPC_ATTEMPTS_PER_RANGE = 8
TRANSIENT_FAILURE_WINDOW = 32
TRANSIENT_FAILURE_WINDOW_THRESHOLD = 3
TRANSIENT_FAILURE_STREAK_THRESHOLD = 2
SUCCESS_STREAK_FOR_RECOVERY = 64


@dataclass(frozen=True, slots=True)
class TimestampRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("timestamp range end must be greater than start")


@dataclass(frozen=True, slots=True)
class BlockRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("block range end must be greater than or equal to start")

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class BlockPullPlan:
    window: TimestampRange
    block_range: BlockRange
    expected_rows: int
    expected_files: int


@dataclass(frozen=True, slots=True)
class BlockHeader:
    number: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class AcquisitionRuntimeSnapshot:
    configured_batch_size: int
    final_batch_size: int
    min_batch_size: int
    configured_concurrency: int
    final_concurrency: int
    concurrency_rungs: tuple[int, ...]
    oversize_error_count: int
    transient_error_count: int
    oversize_backoffs: int
    transient_backoffs: int
    concurrency_recoveries: int


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


@dataclass(slots=True)
class RpcController:
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
    def from_config(cls, config: AcquisitionConfig) -> RpcController:
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


@dataclass(slots=True)
class Web3BlockClient:
    provider: ProviderSpec
    chain: ChainSpec
    _web3: AsyncWeb3 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._web3 = build_async_web3(self.provider, self.chain)

    async def close(self) -> None:
        disconnect = getattr(self._web3.provider, "disconnect", None)
        if disconnect is None:
            return

        result = disconnect()
        if inspect.isawaitable(result):
            await result

    async def _get_block(self, block_number: int) -> BlockHeader:
        return self._header_from_raw_block(await self._raw_block_payload(block_number))

    async def _get_latest_block(self) -> BlockHeader:
        return self._header_from_raw_block(await self._raw_block_payload("latest"))

    async def find_first_block_at_or_after(self, timestamp: int) -> int:
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")

        latest_block = await self._get_latest_block()
        if timestamp > latest_block.timestamp:
            return latest_block.number + 1

        low = 0
        high = latest_block.number
        while low < high:
            middle = (low + high) // 2
            middle_timestamp = (await self._get_block(middle)).timestamp
            if middle_timestamp >= timestamp:
                high = middle
            else:
                low = middle + 1
        return low

    async def resolve_block_range(self, window: TimestampRange) -> BlockRange:
        return BlockRange(
            start=await self.find_first_block_at_or_after(window.start),
            end=await self.find_first_block_at_or_after(window.end),
        )

    async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
        return self.plan_block_range(
            await self.resolve_block_range(window),
            window=window,
            chunk_size=chunk_size,
        )

    def plan_block_range(
        self,
        block_range: BlockRange,
        *,
        window: TimestampRange,
        chunk_size: int,
    ) -> BlockPullPlan:
        expected_rows = block_range.count
        expected_files = 0 if expected_rows == 0 else ceil(expected_rows / chunk_size)
        return BlockPullPlan(
            window=window,
            block_range=block_range,
            expected_rows=expected_rows,
            expected_files=expected_files,
        )

    async def estimate_recent_block_interval(self, sample_size: int = 128) -> float:
        if sample_size <= 1:
            raise ValueError("sample_size must be greater than 1")
        latest = await self._get_latest_block()
        earliest_number = max(0, latest.number - sample_size + 1)
        earliest = await self._get_block(earliest_number)
        observed_blocks = max(1, latest.number - earliest.number)
        observed_seconds = max(1, latest.timestamp - earliest.timestamp)
        return observed_seconds / observed_blocks

    async def get_block_rows(self, block_numbers: list[int]) -> list[CanonicalBlockRow]:
        if not block_numbers:
            return []

        async with self._web3.batch_requests() as batch:
            for block_number in block_numbers:
                batch.add(self._web3.eth.get_block(block_number, False))
            raw_blocks = await batch.async_execute()

        if not isinstance(raw_blocks, list):
            raise TypeError("Expected batch block responses as a list")

        blocks = [self._raw_block_from_response(block) for block in raw_blocks]
        if len(blocks) != len(block_numbers):
            raise RuntimeError(
                f"Expected {len(block_numbers)} block responses, got {len(blocks)}"
            )

        return [build_canonical_block_row(block, self.chain) for block in blocks]

    async def pull_block_range(
        self,
        output_dir: Path,
        *,
        plan: BlockPullPlan,
        chunk_size: int,
        rpc_controller: RpcController,
        reporter: Reporter | None = None,
    ) -> BlockPullPlan:
        reporter = reporter or NullReporter()
        if plan.expected_rows == 0:
            raise ValueError(
                f"No blocks found inside requested block range: {plan.block_range}"
            )

        task_id = reporter.start_task(
            f"pull {self.chain.name} blocks",
            total=plan.expected_rows,
            unit="blocks",
        )
        pending_rows: list[CanonicalBlockRow] = []
        pending_requests: list[_BatchRequest] = []
        in_flight: dict[asyncio.Task[list[CanonicalBlockRow]], _BatchRequest] = {}
        completed_results: dict[int, _CompletedBatch] = {}
        completed = 0
        next_request_start = plan.block_range.start
        next_write_start = plan.block_range.start

        try:
            reporter.update_task(
                task_id,
                completed=0,
                message=self._progress_message(
                    batch_size=rpc_controller.current_batch_size,
                    concurrency=rpc_controller.current_concurrency,
                ),
            )
            while next_write_start < plan.block_range.end:
                while len(in_flight) < rpc_controller.current_concurrency:
                    request = self._next_request(
                        pending_requests,
                        next_request_start=next_request_start,
                        range_end=plan.block_range.end,
                        batch_size=rpc_controller.current_batch_size,
                    )
                    if request is None:
                        break
                    if request.start >= next_request_start:
                        next_request_start = request.end
                    task = asyncio.create_task(
                        self.get_block_rows(list(range(request.start, request.end)))
                    )
                    in_flight[task] = request

                if not in_flight:
                    raise RuntimeError("RPC scheduler stalled before completing the block range")

                done, _ = await asyncio.wait(
                    in_flight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    request = in_flight.pop(task)
                    try:
                        rows = task.result()
                    except Exception as exc:
                        if self._is_oversize_error(exc):
                            next_batch_size = rpc_controller.record_oversize_failure()
                            if (
                                next_batch_size is None
                                or request.size <= rpc_controller.min_batch_size
                            ):
                                raise RuntimeError(
                                    "RPC batch remained too large at the configured minimum "
                                    f"batch size ({rpc_controller.min_batch_size})"
                                ) from exc
                            reporter.update_task(
                                task_id,
                                completed=completed,
                                message=self._progress_message(
                                    batch_size=next_batch_size,
                                    concurrency=rpc_controller.current_concurrency,
                                    note="oversize backoff",
                                ),
                            )
                            for retry_request in self._split_request(
                                request,
                                batch_size=next_batch_size,
                            ):
                                heappush(pending_requests, retry_request)
                            continue

                        if self._is_transient_error(exc):
                            rpc_controller.record_transient_failure()
                            if request.attempts + 1 >= MAX_RPC_ATTEMPTS_PER_RANGE:
                                raise RuntimeError(
                                    f"RPC range {request.start}..{request.end} exceeded "
                                    f"{MAX_RPC_ATTEMPTS_PER_RANGE} transient retry attempts"
                                ) from exc
                            reporter.update_task(
                                task_id,
                                completed=completed,
                                message=self._progress_message(
                                    batch_size=rpc_controller.current_batch_size,
                                    concurrency=rpc_controller.current_concurrency,
                                    note="transient retry",
                                ),
                            )
                            heappush(pending_requests, request.retry())
                            continue

                        raise

                    rpc_controller.record_success()

                    if len(rows) != request.size:
                        raise RuntimeError(
                            f"Expected {request.size} rows for {request.start}..{request.end}, "
                            f"got {len(rows)}"
                        )

                    completed_results[request.start] = _CompletedBatch(
                        start=request.start,
                        end=request.end,
                        rows=rows,
                    )
                    completed += len(rows)
                    reporter.update_task(
                        task_id,
                        completed=completed,
                        message=self._progress_message(
                            batch_size=request.size,
                            concurrency=rpc_controller.current_concurrency,
                        ),
                    )

                    while next_write_start in completed_results:
                        finished_batch = completed_results.pop(next_write_start)
                        pending_rows.extend(finished_batch.rows)
                        next_write_start = finished_batch.end
                        while len(pending_rows) >= chunk_size:
                            self._write_chunk(output_dir, pending_rows[:chunk_size])
                            pending_rows = pending_rows[chunk_size:]

            if pending_rows:
                self._write_chunk(output_dir, pending_rows)

            reporter.finish_task(task_id, silent=True)
            return plan
        finally:
            for task in in_flight:
                task.cancel()
            current_task = asyncio.current_task()
            if in_flight and (current_task is None or current_task.cancelling() == 0):
                await asyncio.gather(*in_flight, return_exceptions=True)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _as_int(value: object) -> int:
        return int(cast(SupportsInt | str | bytes | bytearray, value))

    async def _raw_block_payload(self, block_number: int | Literal["latest"]) -> RpcBlock:
        return self._raw_block_from_response(
            await self._web3.eth.get_block(block_number, False)
        )

    @staticmethod
    def _raw_block_from_response(response: object) -> RpcBlock:
        if not isinstance(response, Mapping):
            raise TypeError(f"Unsupported RPC block payload type: {type(response)!r}")
        return {str(key): value for key, value in response.items()}

    @classmethod
    def _header_from_raw_block(cls, block: RpcBlock) -> BlockHeader:
        return BlockHeader(
            number=cls._as_int(block["number"]),
            timestamp=cls._as_int(block["timestamp"]),
        )

    @staticmethod
    def _error_message(exc: BaseException) -> str:
        parts = [str(exc).lower()]
        if isinstance(exc, Web3RPCError):
            parts.append(exc.message.lower())
            if exc.rpc_response is not None:
                error = exc.rpc_response.get("error")
                if isinstance(error, Mapping):
                    message = error.get("message")
                    if message is not None:
                        parts.append(str(message).lower())
                    code = error.get("code")
                    if code is not None:
                        parts.append(str(code).lower())
        if isinstance(exc, aiohttp.ClientResponseError):
            parts.append(f"status:{exc.status}")
        return " ".join(parts)

    @classmethod
    def _is_oversize_error(cls, exc: BaseException) -> bool:
        message = cls._error_message(exc)
        return "response too large" in message or "batch too large" in message

    @classmethod
    def _is_transient_error(cls, exc: BaseException) -> bool:
        if isinstance(
            exc,
            (
                asyncio.TimeoutError,
                TimeoutError,
                OSError,
                aiohttp.ClientConnectionError,
                aiohttp.ServerTimeoutError,
            ),
        ):
            return True
        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status == 429 or exc.status >= 500

        message = cls._error_message(exc)
        return any(
            token in message
            for token in (
                "timed out",
                "timeout",
                "too many requests",
                "rate limit",
                "temporarily unavailable",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "connection reset",
                "connection aborted",
                "server disconnected",
                "status:429",
                "status:500",
                "status:502",
                "status:503",
                "status:504",
            )
        )

    def _write_chunk(self, output_dir: Path, rows: list[CanonicalBlockRow]) -> Path:
        frame = canonicalize_block_frame(pl.DataFrame(rows))
        start_block = int(frame["block_number"][0])
        end_block = int(frame["block_number"][-1])
        destination = (
            output_dir
            / f"{self.chain.name}__blocks__{start_block}_to_{end_block}.parquet"
        )
        write_block_file(destination, frame)
        return destination

    @staticmethod
    def _progress_message(
        *,
        batch_size: int,
        concurrency: int,
        note: str | None = None,
    ) -> str:
        metrics = f"batch={batch_size} conc={concurrency}"
        if note is None:
            return metrics
        return f"{note} {metrics}"


def evaluation_range(start_timestamp: int, end_timestamp: int) -> TimestampRange:
    return TimestampRange(start=start_timestamp, end=end_timestamp)

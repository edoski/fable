"""RPC block pulling scheduler and retry logic."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import aiohttp
import polars as pl
from web3.exceptions import Web3RPCError

from ...corpus.contract import CanonicalBlockRow, canonicalize_block_frame
from ...corpus.io import write_block_file
from .client import BlockRpcClient
from .controller import RpcController
from .types import BlockPullPlan

MAX_RPC_ATTEMPTS_PER_RANGE = 8


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


async def pull_block_range(
    block_client: BlockRpcClient,
    output_dir: Path,
    *,
    plan: BlockPullPlan,
    chunk_size: int,
    rpc_controller: RpcController,
) -> BlockPullPlan:
    if plan.expected_rows == 0:
        raise ValueError(f"No blocks found inside requested block range: {plan.block_range}")
    pending_rows: list[CanonicalBlockRow] = []
    pending_requests: list[_BatchRequest] = []
    in_flight: dict[asyncio.Task[list[CanonicalBlockRow]], _BatchRequest] = {}
    completed_results: dict[int, _CompletedBatch] = {}
    next_request_start = plan.block_range.start
    next_write_start = plan.block_range.start

    try:
        while next_write_start < plan.block_range.end:
            while len(in_flight) < rpc_controller.current_concurrency:
                request = _next_request(
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
                    block_client.get_block_rows(list(range(request.start, request.end)))
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
                    if _is_oversize_error(exc):
                        next_batch_size = rpc_controller.record_oversize_failure()
                        if next_batch_size is None or request.size <= rpc_controller.min_batch_size:
                            raise RuntimeError(
                                "RPC batch remained too large at the configured minimum "
                                f"batch size ({rpc_controller.min_batch_size})"
                            ) from exc
                        for retry_request in _split_request(request, batch_size=next_batch_size):
                            heappush(pending_requests, retry_request)
                        continue

                    if _is_transient_error(exc):
                        rpc_controller.record_transient_failure()
                        if request.attempts + 1 >= MAX_RPC_ATTEMPTS_PER_RANGE:
                            raise RuntimeError(
                                f"RPC range {request.start}..{request.end} exceeded "
                                f"{MAX_RPC_ATTEMPTS_PER_RANGE} transient retry attempts"
                            ) from exc
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
                while next_write_start in completed_results:
                    finished_batch = completed_results.pop(next_write_start)
                    pending_rows.extend(finished_batch.rows)
                    next_write_start = finished_batch.end
                    while len(pending_rows) >= chunk_size:
                        _write_chunk(
                            output_dir,
                            chain_name=block_client.chain.name,
                            rows=pending_rows[:chunk_size],
                        )
                        pending_rows = pending_rows[chunk_size:]
        if pending_rows:
            _write_chunk(output_dir, chain_name=block_client.chain.name, rows=pending_rows)

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


def _is_oversize_error(exc: BaseException) -> bool:
    message = _error_message(exc)
    return "response too large" in message or "batch too large" in message


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            TimeoutError,
            OSError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
        ),
    ):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or exc.status >= 500

    message = _error_message(exc)
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
            "response payload is not completed",
            "not enough data to satisfy transfer length header",
            "status:429",
            "status:500",
            "status:502",
            "status:503",
            "status:504",
        )
    )


def _write_chunk(output_dir: Path, *, chain_name: str, rows: list[CanonicalBlockRow]) -> Path:
    frame = canonicalize_block_frame(pl.DataFrame(rows))
    start_block = int(frame["block_number"][0])
    end_block = int(frame["block_number"][-1])
    destination = output_dir / f"{chain_name}__blocks__{start_block}_to_{end_block}.parquet"
    write_block_file(destination, frame)
    return destination

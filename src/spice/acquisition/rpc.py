"""web3.py-backed helpers for canonical block acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Literal, SupportsInt, cast

import polars as pl
from web3 import Web3

from ..core.config import ChainConfig, ProviderConfig
from ..core.console import NullReporter, Reporter
from ..data.block_contract import (
    CanonicalBlockRow,
    RpcBlock,
    build_canonical_block_row,
    canonicalize_block_frame,
)
from ..data.io import write_block_file
from .provider import build_web3


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


@dataclass(slots=True)
class Web3BlockClient:
    provider: ProviderConfig
    chain: ChainConfig
    _web3: Web3 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._web3 = build_web3(self.provider, self.chain)

    def _get_block(self, block_number: int) -> BlockHeader:
        return self._header_from_raw_block(self._raw_block_payload(block_number))

    def _get_latest_block(self) -> BlockHeader:
        return self._header_from_raw_block(self._raw_block_payload("latest"))

    def find_first_block_at_or_after(self, timestamp: int) -> int:
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")

        latest_block = self._get_latest_block()
        if timestamp > latest_block.timestamp:
            return latest_block.number + 1

        low = 0
        high = latest_block.number
        while low < high:
            middle = (low + high) // 2
            middle_timestamp = self._get_block(middle).timestamp
            if middle_timestamp >= timestamp:
                high = middle
            else:
                low = middle + 1
        return low

    def resolve_block_range(self, window: TimestampRange) -> BlockRange:
        return BlockRange(
            start=self.find_first_block_at_or_after(window.start),
            end=self.find_first_block_at_or_after(window.end),
        )

    def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
        return self.plan_block_range(
            self.resolve_block_range(window),
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

    def plan_history_window(
        self,
        *,
        end_timestamp: int,
        required_history_blocks: int,
        chunk_size: int,
    ) -> BlockPullPlan:
        if required_history_blocks <= 0:
            raise ValueError("required_history_blocks must be positive")

        evaluation_start_block = self.find_first_block_at_or_after(end_timestamp)
        history_start_block = max(0, evaluation_start_block - required_history_blocks)
        history_start_timestamp = self._get_block(history_start_block).timestamp
        return self.plan_block_range(
            BlockRange(start=history_start_block, end=evaluation_start_block),
            window=TimestampRange(start=history_start_timestamp, end=end_timestamp),
            chunk_size=chunk_size,
        )

    def expand_history_plan(
        self,
        current: BlockPullPlan,
        *,
        observed_row_count: int,
        required_history_blocks: int,
        chunk_size: int,
    ) -> BlockPullPlan:
        missing_blocks = required_history_blocks - observed_row_count
        if missing_blocks <= 0:
            return current

        expanded_start_block = max(
            0,
            current.block_range.start - (missing_blocks + chunk_size),
        )
        expanded_start_timestamp = self._get_block(expanded_start_block).timestamp
        return self.plan_block_range(
            BlockRange(start=expanded_start_block, end=current.block_range.end),
            window=TimestampRange(start=expanded_start_timestamp, end=current.window.end),
            chunk_size=chunk_size,
        )

    def get_block_rows(self, block_numbers: list[int]) -> list[CanonicalBlockRow]:
        if not block_numbers:
            return []

        with self._web3.batch_requests() as batch:
            for block_number in block_numbers:
                batch.add(self._web3.eth.get_block(block_number, False))
            raw_blocks = batch.execute()

        if not isinstance(raw_blocks, list):
            raise TypeError("Expected batch block responses as a list")

        blocks = [self._raw_block_from_response(block) for block in raw_blocks]

        if len(blocks) != len(block_numbers):
            raise RuntimeError(
                f"Expected {len(block_numbers)} block responses, got {len(blocks)}"
            )

        return [build_canonical_block_row(block, self.chain) for block in blocks]

    def pull_block_range(
        self,
        output_dir: Path,
        *,
        plan: BlockPullPlan,
        chunk_size: int,
        rpc_batch_size: int,
        reporter: Reporter | None = None,
    ) -> BlockPullPlan:
        reporter = reporter or NullReporter()
        if plan.expected_rows == 0:
            raise ValueError(
                f"No blocks found inside requested block range: {plan.block_range}"
            )

        task_id = reporter.start_task("pull blocks", total=plan.expected_rows, unit="blocks")
        pending_rows: list[CanonicalBlockRow] = []
        completed = 0

        for batch_start in range(plan.block_range.start, plan.block_range.end, rpc_batch_size):
            batch_end = min(batch_start + rpc_batch_size, plan.block_range.end)
            pending_rows.extend(self.get_block_rows(list(range(batch_start, batch_end))))
            while len(pending_rows) >= chunk_size:
                self._write_chunk(output_dir, pending_rows[:chunk_size])
                pending_rows = pending_rows[chunk_size:]
            completed += batch_end - batch_start
            reporter.update_task(task_id, completed=completed)

        if pending_rows:
            self._write_chunk(output_dir, pending_rows)

        reporter.finish_task(task_id, message=str(output_dir))
        return plan

    @staticmethod
    def _as_int(value: object) -> int:
        return int(cast(SupportsInt | str | bytes | bytearray, value))

    def _raw_block_payload(self, block_number: int | Literal["latest"]) -> RpcBlock:
        return self._raw_block_from_response(self._web3.eth.get_block(block_number, False))

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

    def _write_chunk(self, output_dir: Path, rows: list[CanonicalBlockRow]) -> Path:
        frame = canonicalize_block_frame(pl.DataFrame(rows))
        start_block = int(frame["block_number"][0])
        end_block = int(frame["block_number"][-1])
        destination = (
            output_dir
            / f"{self.chain.name.value}__blocks__{start_block}_to_{end_block}.parquet"
        )
        write_block_file(destination, frame)
        return destination


def evaluation_range(start_timestamp: int, end_timestamp: int) -> TimestampRange:
    return TimestampRange(start=start_timestamp, end=end_timestamp)

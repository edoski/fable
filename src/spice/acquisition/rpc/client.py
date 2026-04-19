"""Web3-backed block client primitives."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from typing import Literal, SupportsInt, cast

from web3 import AsyncWeb3

from ...config import ChainSpec, ProviderSpec
from ...corpus.contract import CanonicalBlockRow, RpcBlock, build_canonical_block_row
from ..provider import build_async_web3
from .types import BlockHeader, BlockPullPlan, BlockRange, TimestampRange


@dataclass(slots=True)
class BlockRpcClient:
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

    @staticmethod
    def _as_int(value: object) -> int:
        return int(cast(SupportsInt | str | bytes | bytearray, value))

    async def _raw_block_payload(self, block_number: int | Literal["latest"]) -> RpcBlock:
        return self._raw_block_from_response(await self._web3.eth.get_block(block_number, False))

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

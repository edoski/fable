"""Web3-backed block client primitives."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, SupportsInt, cast

from web3 import AsyncWeb3
from web3.exceptions import Web3RPCError

from ...config.models import ChainSpec, ResolvedRpcEndpointConfig
from ...corpus.contract import CanonicalBlockRow, RpcBlock, build_canonical_block_row
from .transport import build_async_web3
from .types import BlockHeader, BlockPullPlan, BlockRange, TimestampRange

FEE_HISTORY_REWARD_PERCENTILES = (10.0, 50.0, 90.0)


@dataclass(frozen=True, slots=True)
class _FeeHistoryRow:
    priority_fee_p10: int | None
    priority_fee_p50: int | None
    priority_fee_p90: int | None
    gas_used_ratio: float | None


@dataclass(slots=True)
class BlockRpcClient:
    rpc_endpoint: ResolvedRpcEndpointConfig
    chain: ChainSpec
    _web3: AsyncWeb3 = field(init=False, repr=False)
    _fee_history_unsupported: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._web3 = build_async_web3(self.rpc_endpoint, self.chain)

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

    async def plan_window(self, window: TimestampRange) -> BlockPullPlan:
        return self.plan_block_range(
            await self.resolve_block_range(window),
            window=window,
        )

    def plan_block_range(
        self,
        block_range: BlockRange,
        *,
        window: TimestampRange,
    ) -> BlockPullPlan:
        expected_rows = block_range.count
        return BlockPullPlan(
            window=window,
            block_range=block_range,
            expected_rows=expected_rows,
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

    async def get_block_rows(self, start: int, end: int) -> list[CanonicalBlockRow]:
        if end <= start:
            return []

        async with self._web3.batch_requests() as batch:
            for block_number in range(start, end):
                batch.add(self._web3.eth.get_block(block_number, False))
            raw_blocks = await batch.async_execute()

        if not isinstance(raw_blocks, list):
            raise TypeError("Expected batch block responses as a list")

        blocks = [self._raw_block_from_response(block) for block in raw_blocks]
        if len(blocks) != end - start:
            raise RuntimeError(
                f"Expected {end - start} block responses, got {len(blocks)}"
            )
        self._validate_range_blocks(blocks, start=start, end=end)
        fee_history_rows = await self._fee_history_rows(start, end)
        return [
            build_canonical_block_row(
                block,
                self.chain,
                priority_fee_p10=fee_history.priority_fee_p10,
                priority_fee_p50=fee_history.priority_fee_p50,
                priority_fee_p90=fee_history.priority_fee_p90,
                fee_history_gas_used_ratio=fee_history.gas_used_ratio,
            )
            for block, fee_history in zip(blocks, fee_history_rows, strict=True)
        ]

    @staticmethod
    def _as_int(value: object) -> int:
        return int(cast(SupportsInt | str | bytes | bytearray, value))

    @staticmethod
    def _quantity_int(value: object) -> int:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(cast(SupportsInt | str | bytes | bytearray, value))

    async def _raw_block_payload(self, block_number: int | Literal["latest"]) -> RpcBlock:
        return self._raw_block_from_response(await self._web3.eth.get_block(block_number, False))

    @staticmethod
    def _raw_block_from_response(response: object) -> RpcBlock:
        if not isinstance(response, Mapping):
            raise TypeError(f"Unsupported RPC block payload type: {type(response)!r}")
        return {str(key): value for key, value in response.items()}

    @classmethod
    def _validate_range_blocks(
        cls,
        blocks: list[RpcBlock],
        *,
        start: int,
        end: int,
    ) -> None:
        for index, block in enumerate(blocks):
            expected_block_number = start + index
            actual_block_number = cls._quantity_int(block.get("number"))
            if actual_block_number != expected_block_number:
                raise ValueError(
                    "RPC block response order mismatch: "
                    f"expected block {expected_block_number} at index {index}, "
                    f"got {actual_block_number} for range {start}..{end}"
                )

    async def _fee_history_rows(self, start: int, end: int) -> list[_FeeHistoryRow]:
        block_count = end - start
        if block_count <= 0:
            return []
        if self._fee_history_unsupported:
            return self._null_fee_history_rows(block_count)
        try:
            response = await self._web3.eth.fee_history(
                block_count,
                cast(Any, end - 1),
                list(FEE_HISTORY_REWARD_PERCENTILES),
            )
        except Web3RPCError as exc:
            if self._is_unsupported_fee_history_error(exc):
                self._fee_history_unsupported = True
                return self._null_fee_history_rows(block_count)
            raise
        if not isinstance(response, Mapping):
            raise TypeError(f"Unsupported eth_feeHistory payload type: {type(response)!r}")
        raw_response = {str(key): value for key, value in response.items()}
        oldest_block = self._quantity_int(raw_response.get("oldestBlock"))
        if oldest_block != start:
            raise ValueError(
                f"eth_feeHistory oldestBlock mismatch: expected {start}, got {oldest_block}"
            )
        gas_used_ratios = raw_response.get("gasUsedRatio")
        rewards = raw_response.get("reward")
        if not isinstance(gas_used_ratios, list | tuple):
            raise TypeError("eth_feeHistory gasUsedRatio must be a sequence")
        if not isinstance(rewards, list | tuple):
            raise TypeError("eth_feeHistory reward must be a sequence")
        if len(gas_used_ratios) != block_count:
            raise ValueError(
                "eth_feeHistory gasUsedRatio length mismatch: "
                f"expected {block_count}, got {len(gas_used_ratios)}"
            )
        if len(rewards) != block_count:
            raise ValueError(
                f"eth_feeHistory reward length mismatch: expected {block_count}, got {len(rewards)}"
            )
        rows: list[_FeeHistoryRow] = []
        for index, (reward_row, gas_used_ratio) in enumerate(
            zip(rewards, gas_used_ratios, strict=True)
        ):
            if not isinstance(reward_row, list | tuple):
                raise TypeError(f"eth_feeHistory reward row {index} must be a sequence")
            if len(reward_row) != len(FEE_HISTORY_REWARD_PERCENTILES):
                raise ValueError(
                    f"eth_feeHistory reward row {index} length mismatch: "
                    f"expected {len(FEE_HISTORY_REWARD_PERCENTILES)}, got {len(reward_row)}"
                )
            rows.append(
                _FeeHistoryRow(
                    priority_fee_p10=self._quantity_int(reward_row[0]),
                    priority_fee_p50=self._quantity_int(reward_row[1]),
                    priority_fee_p90=self._quantity_int(reward_row[2]),
                    gas_used_ratio=float(gas_used_ratio),
                )
            )
        return rows

    @staticmethod
    def _null_fee_history_rows(block_count: int) -> list[_FeeHistoryRow]:
        return [
            _FeeHistoryRow(
                priority_fee_p10=None,
                priority_fee_p50=None,
                priority_fee_p90=None,
                gas_used_ratio=None,
            )
            for _ in range(block_count)
        ]

    @staticmethod
    def _is_unsupported_fee_history_error(exc: Web3RPCError) -> bool:
        parts = [str(exc).lower(), exc.message.lower()]
        if exc.rpc_response is not None:
            error = exc.rpc_response.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                message = error.get("message")
                if code is not None:
                    parts.append(str(code).lower())
                if message is not None:
                    parts.append(str(message).lower())
        message = " ".join(parts)
        return any(
            token in message
            for token in (
                "method not found",
                "method not supported",
                "unsupported method",
                "-32601",
            )
        )

    @classmethod
    def _header_from_raw_block(cls, block: RpcBlock) -> BlockHeader:
        return BlockHeader(
            number=cls._as_int(block["number"]),
            timestamp=cls._as_int(block["timestamp"]),
        )

"""Generic acquisition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..corpus.contract import CanonicalBlockRow


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

    def __post_init__(self) -> None:
        if self.block_range.count == 0:
            raise ValueError("block pull plan must contain at least one block")


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


class BlockRowSource(Protocol):
    async def get_block_rows(self, start: int, end: int) -> list[CanonicalBlockRow]: ...


class BlockSource(BlockRowSource, Protocol):
    async def estimate_recent_block_interval(self, sample_size: int = 128) -> float: ...

    async def plan_window(self, window: TimestampRange) -> BlockPullPlan: ...

    def plan_block_range(
        self,
        block_range: BlockRange,
        *,
        window: TimestampRange,
    ) -> BlockPullPlan: ...

    async def close(self) -> None: ...


def evaluation_range(start_timestamp: int, end_timestamp: int) -> TimestampRange:
    return TimestampRange(start=start_timestamp, end=end_timestamp)

"""Shared value types for block-range acquisition."""

from __future__ import annotations

from dataclasses import dataclass


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


def evaluation_range(start_timestamp: int, end_timestamp: int) -> TimestampRange:
    return TimestampRange(start=start_timestamp, end=end_timestamp)

"""Core record types shared across the project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BlockRecord:
    block_number: int
    timestamp: int
    base_fee_per_gas: int
    gas_used: int
    gas_limit: int
    chain_id: int

    @property
    def gas_utilization(self) -> float:
        if self.gas_limit <= 0:
            raise ValueError("gas_limit must be positive to compute gas utilization")
        return self.gas_used / self.gas_limit


@dataclass(slots=True)
class FeatureRow:
    block_number: int
    timestamp: int
    features: list[float]
    log_base_fee: float


@dataclass(slots=True)
class SupervisedExample:
    anchor_block_number: int
    anchor_timestamp: int
    inputs: list[list[float]]
    class_label: int
    target_log_fee: float
    future_log_fees: list[float]
    next_block_log_fee: float
    optimal_log_fee: float

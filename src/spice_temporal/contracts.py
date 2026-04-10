"""Typed contracts for boundary data structures."""

from __future__ import annotations

from typing import NamedTuple, NotRequired, TypeAlias, TypedDict

from torch import Tensor

BlockScalar: TypeAlias = int | str
NullableBlockScalar: TypeAlias = BlockScalar | None


class RawBlockRow(TypedDict):
    block_number: BlockScalar
    timestamp: BlockScalar
    base_fee_per_gas: BlockScalar
    gas_used: BlockScalar
    chain_id: BlockScalar
    gas_limit: NotRequired[NullableBlockScalar]


class EnrichedBlockRow(TypedDict):
    block_number: BlockScalar
    timestamp: BlockScalar
    base_fee_per_gas: BlockScalar
    gas_used: BlockScalar
    chain_id: BlockScalar
    gas_limit: BlockScalar


BlockRow: TypeAlias = RawBlockRow | EnrichedBlockRow


class SequenceBatch(TypedDict):
    inputs: Tensor
    class_label: Tensor
    target_log_fee: Tensor
    action_log_fees: Tensor
    next_block_log_fee: Tensor
    optimal_log_fee: Tensor


class ModelOutputs(NamedTuple):
    logits: Tensor
    fee_hat: Tensor

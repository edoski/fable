from __future__ import annotations

from spice.data.block_contract import (
    BLOCK_COLUMNS,
    BLOCK_SCHEMA,
    CANONICAL_BLOCK_FIELDS,
    build_canonical_block_row,
)
from tests.support import make_chain_config


def test_canonical_block_contract_derives_schema_from_registry() -> None:
    assert tuple(field.name for field in CANONICAL_BLOCK_FIELDS) == tuple(BLOCK_SCHEMA)
    assert {field.name: field.dtype for field in CANONICAL_BLOCK_FIELDS} == BLOCK_SCHEMA
    assert BLOCK_COLUMNS == tuple(field.name for field in CANONICAL_BLOCK_FIELDS)


def test_build_canonical_block_row_uses_contract_and_chain_config() -> None:
    row = build_canonical_block_row(
        {
            "number": 7,
            "timestamp": 1_700_000_007,
            "baseFeePerGas": None,
            "gasUsed": 20_000_007,
            "gasLimit": 30_000_007,
        },
        make_chain_config(),
    )

    assert row == {
        "block_number": 7,
        "timestamp": 1_700_000_007,
        "base_fee_per_gas": 0,
        "gas_used": 20_000_007,
        "chain_id": 1,
        "gas_limit": 30_000_007,
    }

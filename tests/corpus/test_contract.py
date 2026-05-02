from __future__ import annotations

import polars as pl
import pytest

from spice.config.models import ChainSpec
from spice.corpus.contract import build_canonical_block_row, validate_block_frame
from tests.dataset_helpers import make_block_rows


def _chain() -> ChainSpec:
    return ChainSpec.model_validate(
        {
            "name": "ethereum",
            "runtime": {
                "chain_id": 1,
                "uses_poa_extra_data": False,
                "nominal_block_time_seconds": 12.0,
            },
        }
    )


def test_validate_block_frame_rejects_null_required_columns() -> None:
    rows = make_block_rows(3, start_block=1, start_timestamp=1_700_000_000)
    rows[1]["tx_count"] = None

    with pytest.raises(ValueError, match="null required columns: tx_count"):
        validate_block_frame(pl.DataFrame(rows))


def test_build_canonical_block_row_accepts_hex_rpc_quantities() -> None:
    row = build_canonical_block_row(
        {
            "number": "0x64",
            "timestamp": "0x6553f100",
            "baseFeePerGas": "0x3b9aca00",
            "gasUsed": "0x5208",
            "gasLimit": "0x1c9c380",
            "transactions": ["0xabc", "0xdef"],
            "size": "0x400",
            "blobGasUsed": "0x0",
            "excessBlobGas": "0x2",
        },
        _chain(),
        priority_fee_p10=1,
        priority_fee_p50=2,
        priority_fee_p90=4,
    )

    assert row["block_number"] == 100
    assert row["timestamp"] == 1_700_000_000
    assert row["base_fee_per_gas"] == 1_000_000_000
    assert row["gas_used"] == 21_000
    assert row["tx_count"] == 2
    assert row["block_size_bytes"] == 1024
    assert row["blob_gas_used"] == 0
    assert row["excess_blob_gas"] == 2

from __future__ import annotations

import json
from typing import cast

import polars as pl
import pytest
from pydantic import UUID4, TypeAdapter

from fable.addresses import corpus_blocks_path, corpus_json_path
from fable.config import CorpusDefinition, CorpusRequest
from fable.corpus import load_corpus

CORPUS_ID = TypeAdapter(UUID4).validate_python("11111111-1111-4111-8111-111111111111")
OTHER_CORPUS_ID = TypeAdapter(UUID4).validate_python("22222222-2222-4222-8222-222222222222")
BLOCK_SCHEMA = {
    "block_number": pl.Int64,
    "timestamp": pl.Int64,
    "chain_id": pl.Int64,
    "base_fee_per_gas": pl.Int64,
    "gas_used": pl.Int64,
    "gas_limit": pl.Int64,
    "tx_count": pl.Int64,
}


def _valid_document() -> dict[str, object]:
    request = CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=100, last_block=102),
    )
    return {
        "request": request.model_dump(mode="json"),
        "finalized_anchor": {
            "block_number": 103,
            "block_hash": "a" * 64,
        },
    }


def _valid_blocks() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (100, 1_000, 1, 100, 50, 100, 10),
            (101, 1_012, 1, 101, 51, 100, 11),
            (102, 1_024, 1, 102, 52, 100, 12),
        ],
        schema=BLOCK_SCHEMA,
        orient="row",
    )


def _invalidate(
    case: str,
    document: dict[str, object],
    blocks: pl.DataFrame,
) -> pl.DataFrame:
    request = cast(dict[str, object], document["request"])
    anchor = cast(dict[str, object], document["finalized_anchor"])
    if case == "json_shape":
        document["extra"] = True
    elif case == "uuid_association":
        request["corpus_id"] = str(OTHER_CORPUS_ID)
    elif case == "anchor_shape":
        anchor["block_hash"] = "A" * 64
    elif case == "anchor_relation":
        anchor["block_number"] = 101
    elif case == "column_names":
        blocks = blocks.rename({"tx_count": "transactions"})
    elif case == "column_order":
        blocks = blocks.select(
            "timestamp",
            "block_number",
            "chain_id",
            "base_fee_per_gas",
            "gas_used",
            "gas_limit",
            "tx_count",
        )
    elif case == "column_dtype":
        blocks = blocks.with_columns(pl.col("tx_count").cast(pl.Int32))
    elif case == "column_null":
        blocks = blocks.with_columns(
            pl.when(pl.col("block_number") == 101)
            .then(None)
            .otherwise(pl.col("tx_count"))
            .alias("tx_count")
        )
    elif case == "row_count":
        blocks = blocks.head(2)
    elif case == "inclusive_range":
        blocks = blocks.with_columns((pl.col("block_number") + 1).alias("block_number"))
    elif case == "stored_order":
        blocks = pl.concat((blocks.slice(1, 1), blocks.slice(0, 1), blocks.slice(2, 1)))
    elif case == "contiguity":
        blocks = blocks.with_columns(
            pl.when(pl.col("block_number") == 102)
            .then(103)
            .otherwise(pl.col("block_number"))
            .alias("block_number")
        )
    elif case == "chain":
        blocks = blocks.with_columns(
            pl.when(pl.col("block_number") == 101)
            .then(2)
            .otherwise(pl.col("chain_id"))
            .alias("chain_id")
        )
    elif case == "negative_timestamp":
        blocks = blocks.with_columns(
            pl.when(pl.col("block_number") == 100)
            .then(-1)
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
    elif case == "decreasing_timestamp":
        blocks = blocks.with_columns(
            pl.when(pl.col("block_number") == 101)
            .then(999)
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
    elif case == "base_fee":
        blocks = blocks.with_columns(pl.lit(0, dtype=pl.Int64).alias("base_fee_per_gas"))
    elif case == "gas_limit":
        blocks = blocks.with_columns(pl.lit(0, dtype=pl.Int64).alias("gas_limit"))
    elif case == "negative_gas_used":
        blocks = blocks.with_columns(pl.lit(-1, dtype=pl.Int64).alias("gas_used"))
    elif case == "gas_used_above_limit":
        blocks = blocks.with_columns((pl.col("gas_limit") + 1).alias("gas_used"))
    elif case == "tx_count":
        blocks = blocks.with_columns(pl.lit(-1, dtype=pl.Int64).alias("tx_count"))
    else:
        raise AssertionError(f"unknown invalid case: {case}")
    return blocks


@pytest.mark.parametrize(
    "case",
    (
        "json_shape",
        "uuid_association",
        "anchor_shape",
        "anchor_relation",
        "column_names",
        "column_order",
        "column_dtype",
        "column_null",
        "row_count",
        "inclusive_range",
        "stored_order",
        "contiguity",
        "chain",
        "negative_timestamp",
        "decreasing_timestamp",
        "base_fee",
        "gas_limit",
        "negative_gas_used",
        "gas_used_above_limit",
        "tx_count",
    ),
)
def test_load_corpus_rejects_invalid_canonical_facts(tmp_path, case: str) -> None:
    document = _valid_document()
    blocks = _invalidate(case, document, _valid_blocks())
    corpus_json_path(tmp_path, CORPUS_ID).parent.mkdir(parents=True)
    corpus_json_path(tmp_path, CORPUS_ID).write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    blocks.write_parquet(corpus_blocks_path(tmp_path, CORPUS_ID))

    with pytest.raises(ValueError):
        load_corpus(tmp_path, CORPUS_ID)

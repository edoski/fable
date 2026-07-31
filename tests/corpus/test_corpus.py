from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fable.addresses import corpus_blocks_path, corpus_json_path
from fable.config import CorpusDefinition, CorpusRequest
from fable.corpus import load_corpus_blocks

CORPUS_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_CORPUS_ID = UUID("22222222-2222-4222-8222-222222222222")
BLOCK_SCHEMA = {
    "block_number": pl.Int64,
    "timestamp": pl.Int64,
    "chain_id": pl.Int64,
    "base_fee_per_gas": pl.Int64,
    "gas_used": pl.Int64,
    "gas_limit": pl.Int64,
    "tx_count": pl.Int64,
    "effective_priority_fee_per_gas_p50": pl.Int64,
    "effective_priority_fee_per_gas_p90": pl.Int64,
}


def _request() -> CorpusRequest:
    return CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=100, last_block=102),
    )


def _valid_document() -> dict[str, object]:
    return {
        "request": _request().model_dump(mode="json"),
        "finalized_anchor": {"block_number": 103, "block_hash": "a" * 64},
    }


def _valid_blocks() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (100, 1_000, 1, 100, 50, 100, 10, 1, 2),
            (101, 1_012, 1, 101, 51, 100, 11, 2, 4),
            (102, 1_024, 1, 102, 52, 100, 12, 0, 0),
        ],
        schema=BLOCK_SCHEMA,
        orient="row",
    )


def _write_corpus(root: Path, document: dict[str, object], blocks: pl.DataFrame) -> None:
    corpus_json_path(root, CORPUS_ID).parent.mkdir(parents=True)
    corpus_json_path(root, CORPUS_ID).write_text(json.dumps(document), encoding="utf-8")
    blocks.write_parquet(corpus_blocks_path(root, CORPUS_ID))


def test_load_corpus_blocks_reads_one_valid_canonical_pair(tmp_path) -> None:
    blocks = _valid_blocks()
    _write_corpus(tmp_path, _valid_document(), blocks)

    loaded = load_corpus_blocks(tmp_path, CORPUS_ID)

    assert_frame_equal(loaded.to_polars(), blocks)


def test_load_corpus_blocks_rejects_a_mismatched_request_uuid(tmp_path) -> None:
    document = _valid_document()
    request = document["request"]
    assert isinstance(request, dict)
    request["corpus_id"] = str(OTHER_CORPUS_ID)
    _write_corpus(tmp_path, document, _valid_blocks())

    with pytest.raises(ValueError, match="UUID"):
        load_corpus_blocks(tmp_path, CORPUS_ID)

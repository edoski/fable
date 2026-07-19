from __future__ import annotations

import json

import polars as pl
from polars.testing import assert_frame_equal
from pydantic import UUID4, TypeAdapter

from spice.addresses import corpus_blocks_path, corpus_json_path
from spice.config import CorpusDefinition, CorpusRequest
from spice.corpus import Corpus, FinalizedAnchor, load_corpus

CORPUS_ID = TypeAdapter(UUID4).validate_python("11111111-1111-4111-8111-111111111111")
BLOCK_SCHEMA = {
    "block_number": pl.Int64,
    "timestamp": pl.Int64,
    "chain_id": pl.Int64,
    "base_fee_per_gas": pl.Int64,
    "gas_used": pl.Int64,
    "gas_limit": pl.Int64,
    "tx_count": pl.Int64,
}


def test_load_corpus_reads_one_valid_canonical_pair(tmp_path) -> None:
    request = CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=100, last_block=101),
    )
    anchor = {
        "block_number": 102,
        "block_hash": "a" * 64,
    }
    blocks = pl.DataFrame(
        [
            (100, 1_700_000_000, 1, 1_000_000_000, 15_000_000, 30_000_000, 100),
            (101, 1_700_000_012, 1, 1_000_001_000, 16_000_000, 30_000_000, 101),
        ],
        schema=BLOCK_SCHEMA,
        orient="row",
    )
    corpus_json_path(tmp_path, CORPUS_ID).parent.mkdir(parents=True)
    corpus_json_path(tmp_path, CORPUS_ID).write_text(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "finalized_anchor": anchor,
            }
        ),
        encoding="utf-8",
    )
    blocks.write_parquet(corpus_blocks_path(tmp_path, CORPUS_ID))

    corpus = load_corpus(tmp_path, CORPUS_ID)

    assert isinstance(corpus, Corpus)
    assert corpus.request == request
    assert corpus.finalized_anchor == FinalizedAnchor(**anchor)
    assert_frame_equal(corpus.blocks, blocks)

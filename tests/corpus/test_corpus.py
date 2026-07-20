from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import UUID

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fable.addresses import corpus_blocks_path, corpus_json_path
from fable.config import CorpusDefinition, CorpusRequest
from fable.corpus import BlockFrame, Corpus, FinalizedAnchor, load_corpus

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
}


def _request() -> CorpusRequest:
    return CorpusRequest(
        corpus_id=CORPUS_ID,
        definition=CorpusDefinition(chain_id=1, first_block=100, last_block=102),
    )


def _valid_document() -> dict[str, object]:
    return {
        "request": _request().model_dump(mode="json"),
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


def _write_corpus(
    root: Path,
    document: dict[str, object],
    blocks: pl.DataFrame,
) -> None:
    corpus_json_path(root, CORPUS_ID).parent.mkdir(parents=True)
    corpus_json_path(root, CORPUS_ID).write_text(json.dumps(document), encoding="utf-8")
    blocks.write_parquet(corpus_blocks_path(root, CORPUS_ID))


def test_load_corpus_reads_one_valid_canonical_pair(tmp_path) -> None:
    blocks = _valid_blocks()
    _write_corpus(tmp_path, _valid_document(), blocks)

    corpus = load_corpus(tmp_path, CORPUS_ID)

    assert corpus.request == _request()
    assert corpus.finalized_anchor == FinalizedAnchor(block_number=103, block_hash="a" * 64)
    assert_frame_equal(corpus.blocks.to_polars(), blocks)


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
    elif case == "corrupt_blocks":
        blocks = blocks.with_columns(pl.lit(0, dtype=pl.Int64).alias("base_fee_per_gas"))
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
        "corrupt_blocks",
    ),
)
def test_load_corpus_rejects_invalid_canonical_facts(tmp_path, case: str) -> None:
    document = _valid_document()
    blocks = _invalidate(case, document, _valid_blocks())
    _write_corpus(tmp_path, document, blocks)

    with pytest.raises(ValueError):
        load_corpus(tmp_path, CORPUS_ID)


def test_corpus_requires_block_definition_to_match_request() -> None:
    blocks = BlockFrame(_valid_blocks(), _request().definition)
    other_request = _request().model_copy(
        update={"definition": CorpusDefinition(chain_id=1, first_block=99, last_block=101)}
    )

    with pytest.raises(ValueError, match="definition must match"):
        Corpus(
            request=other_request,
            finalized_anchor=FinalizedAnchor(block_number=103, block_hash="a" * 64),
            blocks=blocks,
        )

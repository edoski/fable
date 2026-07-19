"""Validation for one canonical Corpus candidate."""

from __future__ import annotations

from .contract import _BLOCK_SCHEMA, Corpus


def _validate_corpus_candidate(corpus: Corpus) -> None:
    definition = corpus.request.definition
    anchor = corpus.finalized_anchor
    blocks = corpus.blocks

    if anchor.block_number < definition.last_block:
        raise ValueError("Finalized anchor precedes the requested last block")
    if blocks.schema != _BLOCK_SCHEMA:
        raise ValueError(
            f"Corpus block schema must be exactly {_BLOCK_SCHEMA}, got {blocks.schema}"
        )
    null_columns = [column for column in blocks.columns if blocks[column].null_count() > 0]
    if null_columns:
        raise ValueError("Corpus block columns must be non-null: " + ", ".join(null_columns))

    expected_count = definition.last_block - definition.first_block + 1
    if blocks.height != expected_count:
        raise ValueError(f"Corpus row count must be {expected_count}, got {blocks.height}")
    block_numbers = blocks["block_number"]
    if (
        int(block_numbers[0]) != definition.first_block
        or int(block_numbers[-1]) != definition.last_block
    ):
        raise ValueError("Corpus block range does not match the request")
    if blocks.height > 1 and not (block_numbers.diff().drop_nulls() == 1).all():
        raise ValueError("Corpus blocks must be contiguous in stored order")
    if not (blocks["chain_id"] == definition.chain_id).all():
        raise ValueError("Corpus chain_id does not match the request")

    timestamps = blocks["timestamp"]
    if not (timestamps >= 0).all():
        raise ValueError("Corpus timestamps must be nonnegative")
    if blocks.height > 1 and not (timestamps.diff().drop_nulls() >= 0).all():
        raise ValueError("Corpus timestamps must be nondecreasing")
    if not (blocks["base_fee_per_gas"] > 0).all():
        raise ValueError("Corpus base_fee_per_gas values must be positive")
    if not (blocks["gas_limit"] > 0).all():
        raise ValueError("Corpus gas_limit values must be positive")
    if not (blocks["gas_used"] >= 0).all() or not (blocks["gas_used"] <= blocks["gas_limit"]).all():
        raise ValueError("Corpus gas_used values must be between zero and gas_limit")
    if not (blocks["tx_count"] >= 0).all():
        raise ValueError("Corpus tx_count values must be nonnegative")

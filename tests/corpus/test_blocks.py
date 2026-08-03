from __future__ import annotations

import polars as pl
import pytest

from kairos.config import CorpusDefinition
from kairos.corpus import BlockFrame


def _definition(first_block: int = 100, last_block: int = 104) -> CorpusDefinition:
    return CorpusDefinition(chain_id=1, first_block=first_block, last_block=last_block)


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (100, 1_000, 1, 100, 50, 100, 10, 0, 0),
            (101, 1_012, 1, 101, 51, 100, 11, 1, 2),
            (102, 1_012, 1, 102, 52, 100, 12, 2, 4),
            (103, 1_024, 1, 103, 53, 100, 13, 3, 6),
            (104, 1_036, 1, 104, 54, 100, 14, 4, 8),
        ],
        schema={
            "block_number": pl.Int64,
            "timestamp": pl.Int64,
            "chain_id": pl.Int64,
            "base_fee_per_gas": pl.Int64,
            "gas_used": pl.Int64,
            "gas_limit": pl.Int64,
            "tx_count": pl.Int64,
            "effective_priority_fee_per_gas_p50": pl.Int64,
            "effective_priority_fee_per_gas_p90": pl.Int64,
        },
        orient="row",
    )


def test_block_frame_requires_the_canonical_schema() -> None:
    frame = _valid_frame()
    reordered = frame.select(
        "timestamp", *[column for column in frame.columns if column != "timestamp"]
    )

    with pytest.raises(ValueError, match="schema"):
        BlockFrame(reordered, _definition())


@pytest.mark.parametrize(
    ("first_block", "last_block", "expected"),
    [
        pytest.param(100, 100, [100], id="first"),
        pytest.param(101, 103, [101, 102, 103], id="middle"),
        pytest.param(104, 104, [104], id="last"),
    ],
)
def test_select_range_returns_exact_inclusive_block_range(
    first_block: int, last_block: int, expected: list[int]
) -> None:
    selected = BlockFrame(_valid_frame(), _definition()).select_range(first_block, last_block)

    assert selected.definition == _definition(first_block, last_block)
    assert selected.to_polars()["block_number"].to_list() == expected


@pytest.mark.parametrize(
    ("first_block", "last_block"),
    [
        pytest.param(102, 101, id="inverted"),
        pytest.param(99, 101, id="before-definition"),
        pytest.param(103, 105, id="after-definition"),
    ],
)
def test_select_range_rejects_invalid_bounds(first_block: int, last_block: int) -> None:
    with pytest.raises(ValueError, match="range"):
        BlockFrame(_valid_frame(), _definition()).select_range(first_block, last_block)


def test_select_range_isolates_selected_frame_from_mutation() -> None:
    source = _valid_frame()
    blocks = BlockFrame(source, _definition())
    selected = blocks.select_range(101, 103)

    source[1, "base_fee_per_gas"] = 999
    returned = selected.to_polars()
    returned[0, "base_fee_per_gas"] = 888

    assert blocks.to_polars()["base_fee_per_gas"].to_list() == [100, 101, 102, 103, 104]
    assert selected.to_polars()["base_fee_per_gas"].to_list() == [101, 102, 103]

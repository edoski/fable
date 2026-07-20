from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fable.config import CorpusDefinition
from fable.corpus import BlockFrame


def _definition(first_block: int = 100, last_block: int = 104) -> CorpusDefinition:
    return CorpusDefinition(chain_id=1, first_block=first_block, last_block=last_block)


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (100, 1_000, 1, 100, 50, 100, 10),
            (101, 1_012, 1, 101, 51, 100, 11),
            (102, 1_012, 1, 102, 52, 100, 12),
            (103, 1_024, 1, 103, 53, 100, 13),
            (104, 1_036, 1, 104, 54, 100, 14),
        ],
        schema={
            "block_number": pl.Int64,
            "timestamp": pl.Int64,
            "chain_id": pl.Int64,
            "base_fee_per_gas": pl.Int64,
            "gas_used": pl.Int64,
            "gas_limit": pl.Int64,
            "tx_count": pl.Int64,
        },
        orient="row",
    )


def test_block_frame_owns_one_valid_canonical_frame() -> None:
    frame = _valid_frame()

    blocks = BlockFrame(frame, _definition())

    assert blocks.definition == _definition()
    assert_frame_equal(blocks.to_polars(), frame)


def _replace(column: str, row: int, value: int | None) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return lambda frame: frame.with_columns(
        pl.when(pl.int_range(pl.len()) == row).then(value).otherwise(pl.col(column)).alias(column)
    )


def _reorder(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("timestamp", *[column for column in frame.columns if column != "timestamp"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda frame: frame.rename({"tx_count": "transactions"}), "schema", id="names"
        ),
        pytest.param(_reorder, "schema", id="order"),
        pytest.param(
            lambda frame: frame.with_columns(pl.col("tx_count").cast(pl.Int32)),
            "schema",
            id="dtype",
        ),
        pytest.param(_replace("tx_count", 1, None), "non-null", id="null"),
        pytest.param(lambda frame: frame.head(4), "row count", id="count"),
        pytest.param(
            lambda frame: frame.with_columns(pl.col("block_number") + 1),
            "Block numbers",
            id="range",
        ),
        pytest.param(_replace("block_number", 2, 101), "Block numbers", id="block-order"),
        pytest.param(_replace("chain_id", 2, 2), "chain_id", id="chain"),
        pytest.param(_replace("timestamp", 0, -1), "nonnegative", id="timestamp-negative"),
        pytest.param(_replace("timestamp", 2, 999), "nondecreasing", id="timestamp-order"),
        pytest.param(_replace("base_fee_per_gas", 1, 0), "base_fee_per_gas", id="fee"),
        pytest.param(_replace("gas_limit", 1, 0), "gas_limit", id="limit"),
        pytest.param(_replace("gas_used", 1, -1), "gas_used", id="used-gas-negative"),
        pytest.param(_replace("gas_used", 1, 101), "gas_used", id="used-gas-above-limit"),
        pytest.param(_replace("tx_count", 1, -1), "tx_count", id="transactions"),
    ],
)
def test_block_frame_rejects_invalid_owned_facts(
    mutate: Callable[[pl.DataFrame], pl.DataFrame],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        BlockFrame(mutate(_valid_frame()), _definition())


@pytest.mark.parametrize(
    ("first_block", "last_block", "expected"),
    [
        pytest.param(100, 100, [100], id="first"),
        pytest.param(101, 103, [101, 102, 103], id="middle"),
        pytest.param(104, 104, [104], id="last"),
    ],
)
def test_select_range_returns_exact_inclusive_block_range(
    first_block: int,
    last_block: int,
    expected: list[int],
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


def test_block_frame_isolates_owned_and_returned_frames_from_mutation() -> None:
    source = _valid_frame()
    blocks = BlockFrame(source, _definition())

    source[0, "base_fee_per_gas"] = 999
    returned = blocks.to_polars()
    returned[1, "base_fee_per_gas"] = 888

    assert blocks.to_polars()["base_fee_per_gas"].to_list() == [100, 101, 102, 103, 104]

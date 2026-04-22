from __future__ import annotations

import polars as pl

from spice.reconstruction.models import SplitCandidate
from spice.reconstruction.splits import apply_split_candidate


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(10)),
            "minBlock": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        }
    )


def test_apply_split_candidate_global_before_split_drops_once() -> None:
    split = apply_split_candidate(
        _frame(),
        candidate=SplitCandidate(0.70, 0.15, "global_before_split"),
        warmup_rows=2,
    )

    assert split.rows_dropped == 2
    assert split.train["timestamp"].to_list()[0] == 2


def test_apply_split_candidate_per_split_drops_each_partition() -> None:
    split = apply_split_candidate(
        _frame(),
        candidate=SplitCandidate(0.60, 0.20, "per_split"),
        warmup_rows=1,
    )

    assert split.rows_dropped == 3
    assert split.train["timestamp"].to_list()[0] == 1
    assert split.validation["timestamp"].to_list()[0] == 7

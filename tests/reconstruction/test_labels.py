from __future__ import annotations

import polars as pl

from spice.reconstruction.labels import materialize_candidate_labels
from spice.reconstruction.models import LabelCandidate


def _blocks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "block_number": [100, 101, 102, 103],
            "timestamp": [0, 10, 20, 30],
            "base_fee_per_gas": [5.0, 5.0, 4.0, 7.0],
        }
    )


def test_materialize_candidate_labels_can_include_current_block() -> None:
    labels = materialize_candidate_labels(
        _blocks(),
        delay_seconds=0,
        candidate=LabelCandidate(
            include_current_block=True,
            deadline_inclusive=True,
            tie_break="earliest",
            tail_policy="drop",
        ),
    )

    assert labels.rows_kept == 4
    assert labels.table["minBlock"].to_list() == [0, 0, 0, 0]
    assert labels.table["minBaseFee"].to_list() == [5.0, 5.0, 4.0, 7.0]


def test_materialize_candidate_labels_respects_latest_tie_break() -> None:
    labels = materialize_candidate_labels(
        _blocks(),
        delay_seconds=20,
        candidate=LabelCandidate(
            include_current_block=False,
            deadline_inclusive=True,
            tie_break="latest",
            tail_policy="drop",
        ),
    )

    assert labels.rows_kept == 2
    assert labels.table["minBlock"].to_list() == [2, 1]
    assert labels.table["minBaseFee"].to_list() == [4.0, 4.0]

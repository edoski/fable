"""Candidate label generation from raw reference block streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .models import LabelCandidate


@dataclass(frozen=True, slots=True)
class MaterializedLabels:
    table: pl.DataFrame
    rows_total: int
    rows_kept: int
    rows_dropped: int


def generate_label_candidates() -> tuple[LabelCandidate, ...]:
    return (
        LabelCandidate(True, True, "earliest", "drop"),
        LabelCandidate(True, True, "earliest", "clip"),
        LabelCandidate(True, True, "latest", "drop"),
        LabelCandidate(True, True, "latest", "clip"),
        LabelCandidate(False, True, "earliest", "drop"),
        LabelCandidate(False, True, "earliest", "clip"),
        LabelCandidate(False, True, "latest", "drop"),
        LabelCandidate(False, True, "latest", "clip"),
        LabelCandidate(True, False, "earliest", "drop"),
        LabelCandidate(True, False, "earliest", "clip"),
        LabelCandidate(False, False, "earliest", "drop"),
        LabelCandidate(False, False, "earliest", "clip"),
    )


def materialize_candidate_labels(
    blocks: pl.DataFrame,
    *,
    delay_seconds: int,
    candidate: LabelCandidate,
) -> MaterializedLabels:
    timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
    base_fees = blocks["base_fee_per_gas"].to_numpy().astype(np.float64, copy=False)
    n_rows = int(blocks.height)
    deadlines = timestamps.astype(np.float64) + float(delay_seconds)
    search_side = "right" if candidate.deadline_inclusive else "left"
    end_rows = np.searchsorted(
        timestamps.astype(np.float64, copy=False),
        deadlines,
        side=search_side,
    ).astype(np.int64, copy=False) - 1

    min_block = np.full(n_rows, -1, dtype=np.int64)
    min_base_fee = np.full(n_rows, np.nan, dtype=np.float64)
    valid = np.zeros(n_rows, dtype=np.bool_)

    last_timestamp = int(timestamps[-1]) if timestamps.size else 0
    for anchor in range(n_rows):
        start = anchor if candidate.include_current_block else anchor + 1
        if start >= n_rows:
            continue
        deadline = int(deadlines[anchor])
        if candidate.tail_policy == "drop" and last_timestamp < deadline:
            continue
        end = int(end_rows[anchor])
        if end < start:
            continue
        window = base_fees[start : end + 1]
        if window.size == 0:
            continue
        best_local = int(np.argmin(window))
        if candidate.tie_break == "latest":
            best_local = int(window.size - 1 - np.argmin(window[::-1]))
        selected = start + best_local
        min_block[anchor] = selected - anchor
        min_base_fee[anchor] = float(base_fees[selected])
        valid[anchor] = True

    labeled = blocks.with_columns(
        pl.Series("minBlock", min_block),
        pl.Series("minBaseFee", min_base_fee),
        pl.Series("label_valid", valid),
    )
    filtered = labeled.filter(pl.col("label_valid")).drop("label_valid")
    return MaterializedLabels(
        table=filtered,
        rows_total=n_rows,
        rows_kept=int(filtered.height),
        rows_dropped=n_rows - int(filtered.height),
    )

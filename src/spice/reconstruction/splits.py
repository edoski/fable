"""Chronological split candidates for reference reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .models import SplitCandidate


@dataclass(frozen=True, slots=True)
class MaterializedSplit:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    rows_total: int
    rows_kept: int
    rows_dropped: int


def generate_split_candidates() -> tuple[SplitCandidate, ...]:
    base = (
        SplitCandidate(0.70, 0.15, "global_before_split"),
        SplitCandidate(0.70, 0.15, "per_split"),
        SplitCandidate(0.74, 0.13, "global_before_split"),
        SplitCandidate(0.74, 0.13, "per_split"),
        SplitCandidate(0.80, 0.10, "global_before_split"),
        SplitCandidate(0.80, 0.10, "per_split"),
    )
    return base


def apply_split_candidate(
    frame: pl.DataFrame,
    *,
    candidate: SplitCandidate,
    warmup_rows: int,
) -> MaterializedSplit:
    if frame.height < 3:
        raise ValueError("split candidate requires at least three rows")
    if warmup_rows < 0:
        raise ValueError("warmup_rows must be non-negative")

    rows_total = int(frame.height)
    rows_dropped = 0
    working = frame.sort("timestamp") if "timestamp" in frame.columns else frame
    if candidate.warmup_mode == "global_before_split":
        if working.height <= warmup_rows + 2:
            raise ValueError("global warmup leaves too few rows for train/validation/test")
        working = working.slice(warmup_rows)
        rows_dropped += warmup_rows

    train_end, validation_end = _split_row_bounds(
        int(working.height),
        train_fraction=candidate.train_fraction,
        validation_fraction=candidate.validation_fraction,
    )
    train = working.slice(0, train_end)
    validation = working.slice(train_end, validation_end - train_end)
    test = working.slice(validation_end)

    if candidate.warmup_mode == "per_split":
        train, dropped_train = _drop_split_warmup(train, warmup_rows)
        validation, dropped_validation = _drop_split_warmup(validation, warmup_rows)
        test, dropped_test = _drop_split_warmup(test, warmup_rows)
        rows_dropped += dropped_train + dropped_validation + dropped_test

    if train.height == 0 or validation.height == 0 or test.height == 0:
        raise ValueError("split candidate produced an empty split after warmup")

    rows_kept = int(train.height + validation.height + test.height)
    return MaterializedSplit(
        train=train,
        validation=validation,
        test=test,
        rows_total=rows_total,
        rows_kept=rows_kept,
        rows_dropped=rows_dropped,
    )


def _split_row_bounds(
    n_rows: int,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    train_end = max(1, min(train_end, n_rows - 2))
    validation_end = max(train_end + 1, min(validation_end, n_rows - 1))
    return train_end, validation_end


def _drop_split_warmup(frame: pl.DataFrame, warmup_rows: int) -> tuple[pl.DataFrame, int]:
    if warmup_rows == 0:
        return frame, 0
    if frame.height <= warmup_rows:
        return frame.clear(), int(frame.height)
    return frame.slice(warmup_rows), warmup_rows

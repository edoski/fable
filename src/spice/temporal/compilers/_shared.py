"""Shared geometry and interval helpers for fixed in-repo problem compilers."""

from __future__ import annotations

import numpy as np

from ...features import FeaturePrerequisites, ResolvedFeatureTable
from ..problem_store import CompiledProblemStore


def summarize_positive_timestamp_delta_seconds(
    feature_table: ResolvedFeatureTable,
    *,
    statistic: object,
    empty_error: str,
) -> float:
    deltas = np.diff(feature_table.series.timestamps.astype(np.int64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        raise ValueError(empty_error)
    statistic_value = _enum_value(statistic)
    if statistic_value == "mean":
        return float(np.mean(positive_deltas))
    return float(np.median(positive_deltas))


def build_timestamp_window_store(
    feature_table: ResolvedFeatureTable,
    *,
    feature_prerequisites: FeaturePrerequisites,
    lookback_seconds: int,
    delay_seconds: int,
    max_candidate_slots: int | None = None,
    requires_post_window_row: bool = False,
) -> CompiledProblemStore:
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")

    timestamps = feature_table.series.timestamps
    if timestamps.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    anchor_candidates = np.arange(timestamps.shape[0], dtype=np.int64)
    context_start_rows = np.searchsorted(
        timestamps,
        timestamps - lookback_seconds,
        side="left",
    ).astype(np.int64, copy=False)
    candidate_start_rows = anchor_candidates
    candidate_end_rows = np.searchsorted(
        timestamps,
        timestamps + delay_seconds,
        side="right",
    ).astype(np.int64, copy=False)
    candidate_counts = candidate_end_rows - candidate_start_rows
    context_history_ready = (
        timestamps[context_start_rows] - timestamps[0]
    ) >= feature_prerequisites.history_seconds
    warmup_ready = context_start_rows >= feature_prerequisites.warmup_rows
    future_ready = candidate_counts > 0
    post_window_ready = (
        candidate_end_rows < timestamps.shape[0]
        if requires_post_window_row
        else np.ones_like(candidate_counts, dtype=np.bool_)
    )
    valid_anchor_mask = context_history_ready & warmup_ready & future_ready & post_window_ready
    anchor_rows = anchor_candidates[valid_anchor_mask].astype(np.int64, copy=False)
    if anchor_rows.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    selected_context_starts = context_start_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_starts = candidate_start_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_ends = candidate_end_rows[anchor_rows].astype(np.int64, copy=False)
    selected_candidate_counts = selected_candidate_ends - selected_candidate_starts
    resolved_max_candidate_slots = (
        int(selected_candidate_counts.max())
        if max_candidate_slots is None
        else int(max_candidate_slots)
    )
    if resolved_max_candidate_slots <= 0:
        raise ValueError("max_candidate_slots must be positive")

    return CompiledProblemStore(
        feature_matrix=feature_table.feature_matrix,
        log_base_fees=feature_table.series.log_base_fees,
        timestamps=timestamps,
        anchor_rows=anchor_rows,
        context_start_rows=selected_context_starts,
        candidate_start_rows=selected_candidate_starts,
        candidate_end_rows=selected_candidate_ends,
        max_candidate_slots=resolved_max_candidate_slots,
    )


def _enum_value(value: object) -> str:
    resolved = getattr(value, "value", value)
    return str(resolved)

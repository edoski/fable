"""Shared geometry and interval helpers for fixed in-repo problem compilers."""

from __future__ import annotations

import numpy as np

from ...features import FeaturePrerequisites, ResolvedFeatureTable
from ..problem_store import CompiledProblemStore
from ..semantics import ActionSpaceMode, CandidateStartMode


def calibrate_positive_timestamp_delta_seconds(
    feature_table: ResolvedFeatureTable,
    *,
    statistic: object,
    empty_error: str,
) -> float:
    return summarize_positive_timestamp_delta_seconds(
        feature_table,
        statistic=statistic,
        empty_error=empty_error,
    )


def summarize_positive_timestamp_delta_seconds(
    feature_table: ResolvedFeatureTable,
    *,
    statistic: object,
    empty_error: str,
    window_blocks: int | None = None,
    quantile: float | None = None,
) -> float:
    deltas = np.diff(feature_table.series.timestamps.astype(np.int64, copy=False))
    positive_deltas = deltas[deltas > 0]
    if window_blocks is not None:
        if window_blocks <= 0:
            raise ValueError("window_blocks must be positive")
        positive_deltas = positive_deltas[-window_blocks:]
    if positive_deltas.size == 0:
        raise ValueError(empty_error)
    statistic_value = _enum_value(statistic)
    if statistic_value == "mean":
        return float(np.mean(positive_deltas))
    if statistic_value == "quantile":
        if quantile is None or not 0.0 < quantile < 1.0:
            raise ValueError("quantile statistic requires quantile in (0, 1)")
        return float(np.quantile(positive_deltas, quantile))
    return float(np.median(positive_deltas))


def resolve_runtime_interval_seconds(
    *,
    source: object,
    calibrated_interval_seconds: float,
    nominal_block_time_seconds: float | None,
    compiler_label: str,
    interval_label: str,
) -> float:
    if _enum_value(source) == "calibrated":
        return calibrated_interval_seconds
    if nominal_block_time_seconds is None or nominal_block_time_seconds <= 0:
        raise ValueError(
            f"{compiler_label} requires nominal block time for {interval_label} interval resolution"
        )
    return nominal_block_time_seconds


def resolve_interval_estimator_seconds(
    *,
    estimator: object,
    feature_table: ResolvedFeatureTable,
    nominal_block_time_seconds: float | None,
    compiler_label: str,
    interval_label: str,
) -> float:
    estimator_id = _enum_value(getattr(estimator, "id", estimator))
    if estimator_id == "nominal":
        if nominal_block_time_seconds is None or nominal_block_time_seconds <= 0:
            raise ValueError(
                f"{compiler_label} requires nominal block time "
                f"for {interval_label} interval resolution"
            )
        return nominal_block_time_seconds
    if estimator_id != "recent_deltas":
        raise ValueError(f"Unsupported {compiler_label} interval estimator: {estimator_id}")
    recent_deltas_estimator = estimator
    return summarize_positive_timestamp_delta_seconds(
        feature_table,
        statistic=recent_deltas_estimator.statistic,
        quantile=recent_deltas_estimator.quantile,
        window_blocks=int(recent_deltas_estimator.window_blocks),
        empty_error=f"{compiler_label} requires positive timestamp deltas",
    )


def resolve_bootstrap_interval_seconds(
    *,
    estimator: object,
    recent_block_interval_seconds: float | None,
    nominal_block_time_seconds: float | None,
) -> float | None:
    estimator_id = _enum_value(getattr(estimator, "id", estimator))
    if estimator_id == "nominal":
        return nominal_block_time_seconds
    if recent_block_interval_seconds is None or recent_block_interval_seconds <= 0:
        return None
    return recent_block_interval_seconds


def build_timestamp_window_store(
    feature_table: ResolvedFeatureTable,
    *,
    feature_prerequisites: FeaturePrerequisites,
    lookback_seconds: int,
    delay_seconds: int,
    candidate_start_mode: CandidateStartMode,
    action_space_mode: ActionSpaceMode,
    max_candidate_slots: int | None = None,
    fixed_candidate_count: int | None = None,
    fixed_candidate_count_error: str | None = None,
    requires_post_window_row: bool = False,
) -> CompiledProblemStore:
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")
    if fixed_candidate_count is not None and fixed_candidate_count <= 0:
        raise ValueError("fixed_candidate_count must be positive")

    timestamps = feature_table.series.timestamps
    if timestamps.size == 0:
        raise ValueError("Feature table is too short to produce any supervised samples")

    anchor_candidates = np.arange(timestamps.shape[0], dtype=np.int64)
    candidate_start_offset = 0 if candidate_start_mode is CandidateStartMode.CURRENT_ROW else 1
    context_start_rows = np.searchsorted(
        timestamps,
        timestamps - lookback_seconds,
        side="left",
    ).astype(np.int64, copy=False)
    candidate_start_rows = (anchor_candidates + candidate_start_offset).astype(
        np.int64,
        copy=False,
    )
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
    if fixed_candidate_count is not None:
        if (
            action_space_mode is not ActionSpaceMode.FIXED_EX_ANTE
            and np.any(selected_candidate_counts > fixed_candidate_count)
        ):
            raise ValueError(
                fixed_candidate_count_error
                or "Configured fixed candidate count is too small for this dataset"
            )
        resolved_max_candidate_slots = (
            fixed_candidate_count if max_candidate_slots is None else int(max_candidate_slots)
        )
    else:
        resolved_max_candidate_slots = (
            int(selected_candidate_counts.max())
            if max_candidate_slots is None
            else int(max_candidate_slots)
        )
        if (
            action_space_mode is not ActionSpaceMode.FIXED_EX_ANTE
            and np.any(selected_candidate_counts > resolved_max_candidate_slots)
        ):
            raise ValueError("Configured max_candidate_slots is too small for this dataset")
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
        action_space_mode=action_space_mode,
        max_candidate_slots=resolved_max_candidate_slots,
    )


def _enum_value(value: object) -> str:
    resolved = getattr(value, "value", value)
    return str(resolved)

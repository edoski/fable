"""Candidate engineered feature formulas for ambiguous reference columns."""

from __future__ import annotations

import numpy as np
import polars as pl

from .models import FeatureCandidate, FeatureCandidateSummary


def generate_feature_candidates() -> tuple[FeatureCandidate, ...]:
    return (
        FeatureCandidate(
            gas_ratio_mode="derived_percent",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_prev_delta_sign",
        ),
        FeatureCandidate(
            gas_ratio_mode="derived_percent",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_slope_sign_200",
        ),
        FeatureCandidate(
            gas_ratio_mode="derived_unit",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_slope_sign_200",
        ),
        FeatureCandidate(
            gas_ratio_mode="derived_percent",
            time_since_start_mode="elapsed_blocks",
            base_fee_trend_mode="binary_slope_sign_200",
        ),
        FeatureCandidate(
            gas_ratio_mode="derived_percent",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_mean_delta_200",
        ),
        FeatureCandidate(
            gas_ratio_mode="csv_raw",
            time_since_start_mode="elapsed_seconds",
            base_fee_trend_mode="binary_slope_sign_200",
        ),
    )


def materialize_feature_candidate(
    blocks: pl.DataFrame,
    *,
    chain: str,
    delay_seconds: int,
    candidate: FeatureCandidate,
) -> tuple[pl.DataFrame, FeatureCandidateSummary]:
    timestamps = blocks["timestamp"].to_numpy().astype(np.int64, copy=False)
    block_numbers = blocks["block_number"].to_numpy().astype(np.int64, copy=False)
    gas_used = blocks["gas_used"].to_numpy().astype(np.float64, copy=False)
    gas_limit = blocks["gas_limit"].to_numpy().astype(np.float64, copy=False)
    base_fee = blocks["base_fee_per_gas"].to_numpy().astype(np.float64, copy=False)

    gas_ratio = _gas_ratio(
        blocks,
        gas_used=gas_used,
        gas_limit=gas_limit,
        mode=candidate.gas_ratio_mode,
    )
    time_since_start = _time_since_start(
        timestamps=timestamps,
        block_numbers=block_numbers,
        mode=candidate.time_since_start_mode,
    )
    base_fee_trend = _base_fee_trend(base_fee=base_fee, mode=candidate.base_fee_trend_mode)

    materialized = blocks.with_columns(
        pl.Series("gas_ratio", gas_ratio),
        pl.Series("time_since_start", time_since_start),
        pl.Series("base_fee_trend", base_fee_trend),
    )
    feature_score, feature_notes = _feature_candidate_score(
        timestamps=timestamps,
        summary_gas_ratio_max=float(np.nanmax(gas_ratio)),
        time_since_start_last=float(time_since_start[-1]) if time_since_start.size else 0.0,
        base_fee_trend_unique=tuple(
            float(value)
            for value in np.unique(base_fee_trend[~np.isnan(base_fee_trend)])
        ),
        candidate=candidate,
    )
    summary = FeatureCandidateSummary(
        chain=chain,
        delay_seconds=delay_seconds,
        feature_candidate=candidate,
        gas_ratio_min=float(np.nanmin(gas_ratio)),
        gas_ratio_max=float(np.nanmax(gas_ratio)),
        time_since_start_last=float(time_since_start[-1]) if time_since_start.size else 0.0,
        base_fee_trend_unique=tuple(
            float(value)
            for value in np.unique(base_fee_trend[~np.isnan(base_fee_trend)])
        ),
        score=feature_score,
        notes=feature_notes,
    )
    return materialized, summary


def feature_warmup_rows() -> int:
    return 199


def _gas_ratio(
    blocks: pl.DataFrame,
    *,
    gas_used: np.ndarray,
    gas_limit: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "derived_percent":
        return (gas_used / gas_limit) * 100.0
    if mode == "derived_unit":
        return gas_used / gas_limit
    if mode == "csv_raw":
        return blocks["block_usage_ratio"].to_numpy().astype(np.float64, copy=False)
    raise ValueError(f"Unsupported gas_ratio_mode: {mode}")


def _time_since_start(
    *,
    timestamps: np.ndarray,
    block_numbers: np.ndarray,
    mode: str,
) -> np.ndarray:
    if timestamps.size == 0:
        return np.empty(0, dtype=np.float64)
    if mode == "elapsed_seconds":
        return timestamps.astype(np.float64, copy=False) - float(timestamps[0])
    if mode == "elapsed_blocks":
        return block_numbers.astype(np.float64, copy=False) - float(block_numbers[0])
    raise ValueError(f"Unsupported time_since_start_mode: {mode}")


def _base_fee_trend(*, base_fee: np.ndarray, mode: str) -> np.ndarray:
    log_fee = np.log1p(base_fee.astype(np.float64, copy=False))
    if mode == "binary_slope_sign_200":
        slope = _rolling_linear_slope(log_fee, window=200)
        return _binary_sign(slope)
    if mode == "raw_slope_200":
        return _rolling_linear_slope(log_fee, window=200)
    if mode == "binary_mean_delta_200":
        rolling_mean = _rolling_mean(log_fee, window=200)
        return _binary_sign(log_fee - rolling_mean)
    if mode == "binary_prev_delta_sign":
        return _binary_sign(_delta(log_fee))
    raise ValueError(f"Unsupported base_fee_trend_mode: {mode}")


def _rolling_mean(values: np.ndarray, *, window: int) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size < window:
        return result
    kernel = np.ones(window, dtype=np.float64) / float(window)
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def _delta(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size <= 1:
        return result
    result[1:] = np.diff(values)
    return result


def _rolling_linear_slope(values: np.ndarray, *, window: int) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size < window:
        return result
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    for end in range(window - 1, values.shape[0]):
        window_values = values[end - window + 1 : end + 1]
        y_centered = window_values - float(window_values.mean())
        result[end] = float(np.dot(x_centered, y_centered) / denominator)
    return result


def _binary_sign(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = ~np.isnan(values)
    result[valid] = np.where(values[valid] >= 0.0, 1.0, -1.0)
    return result


def _feature_candidate_score(
    *,
    timestamps: np.ndarray,
    summary_gas_ratio_max: float,
    time_since_start_last: float,
    base_fee_trend_unique: tuple[float, ...],
    candidate: FeatureCandidate,
) -> tuple[float, tuple[str, ...]]:
    notes: list[str] = []
    score = 0.0

    if summary_gas_ratio_max <= 2.0:
        score += 5.0
        notes.append(
            "gas_ratio remains unit-scale, but the reference samples show "
            "percentage-scale values"
        )
    else:
        notes.append("gas_ratio matches percentage-scale reference samples")

    reference_span = float(timestamps[-1] - timestamps[0]) if timestamps.size else 0.0
    if reference_span > 0.0:
        time_error = abs(time_since_start_last - reference_span) / reference_span
        score += time_error
        if time_error < 1e-9:
            notes.append("time_since_start matches elapsed-seconds chronology")
        else:
            notes.append("time_since_start diverges from elapsed-seconds chronology")

    if set(base_fee_trend_unique) != {-1.0, 1.0}:
        score += 5.0
        notes.append("base_fee_trend is not binary in {-1, 1}")
    else:
        notes.append("base_fee_trend remains binary in {-1, 1}")

    if candidate.base_fee_trend_mode == "binary_prev_delta_sign":
        score -= 0.25
        notes.append("favored because notebook samples align with previous-delta sign")

    return score, tuple(notes)

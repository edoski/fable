"""Feature normalization utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]


class ScalerStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    means: list[float]
    scales: list[float]


def window_row_multiplicities(
    *,
    context_start_rows: IntVector,
    anchor_rows: IntVector,
    sample_indices: IntVector,
    n_rows: int,
) -> IntVector:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if sample_indices.size == 0:
        raise ValueError("sample_indices must be non-empty")

    counts = np.zeros(n_rows + 1, dtype=np.int64)
    starts = context_start_rows[sample_indices]
    ends = anchor_rows[sample_indices] + 1
    np.add.at(counts, starts, 1)
    np.add.at(counts, ends, -1)
    return np.cumsum(counts[:-1], dtype=np.int64)


def _scaler_stats(means: NDArray[np.float64], scales: NDArray[np.float64]) -> ScalerStats:
    return ScalerStats(
        means=means.tolist(),
        scales=scales.tolist(),
    )


def _safe_scales(variances: NDArray[np.float64]) -> NDArray[np.float64]:
    scales = np.sqrt(np.maximum(variances, 0.0), dtype=np.float64)
    scales[variances <= 0.0] = 1.0
    return scales


def fit_window_weighted_standard_scaler(
    feature_matrix: FloatMatrix,
    *,
    context_start_rows: IntVector,
    anchor_rows: IntVector,
    sample_indices: IntVector,
) -> ScalerStats:
    if feature_matrix.size == 0:
        raise ValueError("feature_matrix must be non-empty")
    multiplicities = window_row_multiplicities(
        context_start_rows=context_start_rows,
        anchor_rows=anchor_rows,
        sample_indices=sample_indices,
        n_rows=int(feature_matrix.shape[0]),
    )
    weights = multiplicities.astype(np.float64, copy=False)
    if float(weights.sum()) <= 0.0:
        raise ValueError("training windows did not cover any feature rows")
    features = feature_matrix.astype(np.float64, copy=False)
    means = np.average(features, axis=0, weights=weights).astype(np.float64, copy=False)
    centered = features - means
    variances = np.average(centered * centered, axis=0, weights=weights).astype(
        np.float64,
        copy=False,
    )
    return _scaler_stats(means, _safe_scales(variances))


def fit_row_standard_scaler(
    feature_matrix: FloatMatrix,
    *,
    context_start_rows: IntVector,
    anchor_rows: IntVector,
    sample_indices: IntVector,
) -> ScalerStats:
    if feature_matrix.size == 0:
        raise ValueError("feature_matrix must be non-empty")
    multiplicities = window_row_multiplicities(
        context_start_rows=context_start_rows,
        anchor_rows=anchor_rows,
        sample_indices=sample_indices,
        n_rows=int(feature_matrix.shape[0]),
    )
    covered_rows = multiplicities > 0
    if not np.any(covered_rows):
        raise ValueError("training windows did not cover any feature rows")
    covered = feature_matrix[covered_rows].astype(np.float64, copy=False)
    means = covered.mean(axis=0, dtype=np.float64)
    variances = covered.var(axis=0, ddof=0, dtype=np.float64)
    return _scaler_stats(means, _safe_scales(variances))


def fit_standard_scaler(
    feature_matrix: FloatMatrix,
    *,
    context_start_rows: IntVector,
    anchor_rows: IntVector,
    sample_indices: IntVector,
) -> ScalerStats:
    return fit_window_weighted_standard_scaler(
        feature_matrix,
        context_start_rows=context_start_rows,
        anchor_rows=anchor_rows,
        sample_indices=sample_indices,
    )


def transform_feature_matrix(feature_matrix: FloatMatrix, scaler: ScalerStats) -> FloatMatrix:
    means = np.asarray(scaler.means, dtype=np.float32)
    scales = np.asarray(scaler.scales, dtype=np.float32)
    safe_scales = np.where(scales > 0.0, scales, np.float32(1.0))
    return ((feature_matrix - means) / safe_scales).astype(np.float32, copy=False)

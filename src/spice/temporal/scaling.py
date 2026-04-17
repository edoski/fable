"""Feature normalization utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import StandardScaler

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


def _scaler_stats_from_estimator(scaler: StandardScaler) -> ScalerStats:
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("StandardScaler did not produce mean and scale statistics")
    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    return ScalerStats(
        means=means.tolist(),
        scales=scales.tolist(),
    )


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
    scaler = StandardScaler()
    scaler.fit(feature_matrix.astype(np.float64, copy=False), sample_weight=weights)
    return _scaler_stats_from_estimator(scaler)


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
    scaler = StandardScaler()
    scaler.fit(feature_matrix[covered_rows].astype(np.float64, copy=False))
    return _scaler_stats_from_estimator(scaler)


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

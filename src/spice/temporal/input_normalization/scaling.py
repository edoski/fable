"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from ..problem_store import CompiledProblemStore

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]


class ScalerStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    means: list[float]
    scales: list[float]


def _scaler_stats(means: NDArray[np.float64], scales: NDArray[np.float64]) -> ScalerStats:
    return ScalerStats(
        means=means.tolist(),
        scales=scales.tolist(),
    )


def _fit_standard_scaler_stats(
    features: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> ScalerStats:
    scaler = StandardScaler()
    scaler.fit(features, sample_weight=sample_weight)
    if scaler.mean_ is None or scaler.scale_ is None:
        raise ValueError("standard scaler did not produce feature statistics")
    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    return _scaler_stats(
        means,
        scales,
    )


def fit_window_weighted_standard_scaler(
    store: CompiledProblemStore,
    *,
    sample_indices: IntVector,
) -> ScalerStats:
    if store.feature_matrix.size == 0:
        raise ValueError("feature_matrix must be non-empty")
    multiplicities = store.context_row_multiplicities(sample_indices)
    weights = multiplicities.astype(np.float64, copy=False)
    if float(weights.sum()) <= 0.0:
        raise ValueError("training windows did not cover any feature rows")
    features = store.feature_matrix.astype(np.float64, copy=False)
    return _fit_standard_scaler_stats(features, sample_weight=weights)


def fit_row_standard_scaler(
    store: CompiledProblemStore,
    *,
    sample_indices: IntVector,
) -> ScalerStats:
    if store.feature_matrix.size == 0:
        raise ValueError("feature_matrix must be non-empty")
    multiplicities = store.context_row_multiplicities(sample_indices)
    covered_rows = multiplicities > 0
    if not np.any(covered_rows):
        raise ValueError("training windows did not cover any feature rows")
    covered = store.feature_matrix[covered_rows].astype(np.float64, copy=False)
    return _fit_standard_scaler_stats(covered)


def transform_feature_matrix(feature_matrix: FloatMatrix, scaler: ScalerStats) -> FloatMatrix:
    means = np.asarray(scaler.means, dtype=np.float32)
    scales = np.asarray(scaler.scales, dtype=np.float32)
    safe_scales = np.where(scales > 0.0, scales, np.float32(1.0))
    return ((feature_matrix - means) / safe_scales).astype(np.float32, copy=False)


def transform_problem_store_features(
    store: CompiledProblemStore,
    scaler: ScalerStats,
) -> CompiledProblemStore:
    return replace(
        store,
        feature_matrix=transform_feature_matrix(store.feature_matrix, scaler),
    )

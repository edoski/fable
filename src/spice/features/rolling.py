"""Time-window Hamilton feature nodes."""

from __future__ import annotations

import numpy as np
from hamilton.function_modifiers import tag
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


def _window_bounds(timestamps: IntVector, *, window_seconds: int) -> IntVector:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    starts = np.searchsorted(
        timestamps,
        timestamps - window_seconds,
        side="left",
    )
    valid = (timestamps - timestamps[0]) >= window_seconds
    return np.where(valid, starts, -1).astype(np.int64, copy=False)


def _rolling_mean(values: FloatVector, starts: IntVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size == 0:
        return result
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    for index, start in enumerate(starts):
        if start < 0:
            continue
        count = index - start + 1
        result[index] = (cumsum[index + 1] - cumsum[start]) / count
    return result


def _rolling_std(values: FloatVector, starts: IntVector) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.size == 0:
        return result
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    square_cumsum = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
    for index, start in enumerate(starts):
        if start < 0:
            continue
        count = index - start + 1
        mean = (cumsum[index + 1] - cumsum[start]) / count
        second_moment = (square_cumsum[index + 1] - square_cumsum[start]) / count
        variance = max(0.0, second_moment - mean * mean)
        result[index] = variance**0.5
    return result


def _rolling_mean_feature(
    values: FloatVector,
    timestamps: IntVector,
    *,
    window_seconds: int,
) -> FloatVector:
    return _rolling_mean(values, _window_bounds(timestamps, window_seconds=window_seconds))


def _rolling_std_feature(
    values: FloatVector,
    timestamps: IntVector,
    *,
    window_seconds: int,
) -> FloatVector:
    return _rolling_std(values, _window_bounds(timestamps, window_seconds=window_seconds))


@tag(spice_kind="feature", spice_history_seconds="60")
def rolling_mean_log_base_fee_60s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(log_base_fee, timestamps, window_seconds=60)


@tag(spice_kind="feature", spice_history_seconds="60")
def rolling_std_log_base_fee_60s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(log_base_fee, timestamps, window_seconds=60)


@tag(spice_kind="feature", spice_history_seconds="60")
def rolling_mean_gas_utilization_60s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(gas_utilization, timestamps, window_seconds=60)


@tag(spice_kind="feature", spice_history_seconds="60")
def rolling_std_gas_utilization_60s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(gas_utilization, timestamps, window_seconds=60)


@tag(spice_kind="feature", spice_history_seconds="300")
def rolling_mean_log_base_fee_300s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(log_base_fee, timestamps, window_seconds=300)


@tag(spice_kind="feature", spice_history_seconds="300")
def rolling_std_log_base_fee_300s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(log_base_fee, timestamps, window_seconds=300)


@tag(spice_kind="feature", spice_history_seconds="300")
def rolling_mean_gas_utilization_300s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(gas_utilization, timestamps, window_seconds=300)


@tag(spice_kind="feature", spice_history_seconds="300")
def rolling_std_gas_utilization_300s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(gas_utilization, timestamps, window_seconds=300)


@tag(spice_kind="feature", spice_history_seconds="600")
def rolling_mean_log_base_fee_600s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(log_base_fee, timestamps, window_seconds=600)


@tag(spice_kind="feature", spice_history_seconds="600")
def rolling_std_log_base_fee_600s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(log_base_fee, timestamps, window_seconds=600)


@tag(spice_kind="feature", spice_history_seconds="600")
def rolling_mean_gas_utilization_600s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_mean_feature(gas_utilization, timestamps, window_seconds=600)


@tag(spice_kind="feature", spice_history_seconds="600")
def rolling_std_gas_utilization_600s(
    gas_utilization: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _rolling_std_feature(gas_utilization, timestamps, window_seconds=600)

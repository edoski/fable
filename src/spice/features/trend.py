"""Time-window trend Hamilton feature nodes."""

from __future__ import annotations

import numpy as np
from hamilton.function_modifiers import tag
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


def _window_bounds(timestamps: IntVector, *, window_seconds: int) -> IntVector:
    starts = np.searchsorted(timestamps, timestamps - window_seconds, side="left")
    valid = (timestamps - timestamps[0]) >= window_seconds
    return np.where(valid, starts, -1).astype(np.int64, copy=False)


def _trend_slope(values: FloatVector, timestamps: IntVector, *, window_seconds: int) -> FloatVector:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    starts = _window_bounds(timestamps, window_seconds=window_seconds)
    for index, start in enumerate(starts):
        if start < 0:
            continue
        window_values = values[start : index + 1]
        window_times = timestamps[start : index + 1].astype(np.float64, copy=False)
        centered_x = window_times - window_times.mean()
        denominator = np.square(centered_x).sum()
        if denominator <= 0.0:
            result[index] = 0.0
            continue
        centered_y = window_values - window_values.mean()
        result[index] = float((centered_x * centered_y).sum() / denominator)
    return result


@tag(spice_kind="feature", spice_history_seconds="600")
def trend_slope_600s(
    log_base_fee: FloatVector,
    timestamps: IntVector,
) -> FloatVector:
    return _trend_slope(log_base_fee, timestamps, window_seconds=600)

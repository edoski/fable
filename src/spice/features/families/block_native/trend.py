"""Block-window trend Hamilton feature nodes.

SPICE operationalizes the paper's underspecified long-horizon trend
feature as a 200-block slope over log base fee.
"""

from __future__ import annotations

import numpy as np
from hamilton.function_modifiers import tag
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]


def _trend_slopes(log_base_fee: FloatVector, *, window: int) -> FloatVector:
    if log_base_fee.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.full(log_base_fee.shape[0], np.nan, dtype=np.float64)
    if log_base_fee.shape[0] < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(log_base_fee, window_shape=window)
    centered_x = np.arange(window, dtype=np.float64) - (window - 1) / 2
    denominator = np.square(centered_x).sum()
    window_means = windows.mean(axis=1, keepdims=True)
    result[window - 1 :] = ((windows - window_means) * centered_x).sum(axis=1) / denominator
    return result


@tag(spice_kind="feature", spice_history_seconds="0", spice_warmup_rows="199")
def trend_slope_200(log_base_fee: FloatVector) -> FloatVector:
    return _trend_slopes(log_base_fee, window=200)

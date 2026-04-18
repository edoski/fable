"""Window-weighted standardization matching the legacy pipeline."""

from __future__ import annotations

from typing import Literal

from ..scaling import fit_window_weighted_standard_scaler
from .base import CompiledInputNormalizationContract, InputNormalizationConfig


class WindowWeightedStandardConfig(InputNormalizationConfig[Literal["window_weighted_standard"]]):
    id: Literal["window_weighted_standard"] = "window_weighted_standard"


def compile_input_normalization(
    config: WindowWeightedStandardConfig,
) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="window_weighted_standard",
        fit_scaler_fn=fit_window_weighted_standard_scaler,
    )

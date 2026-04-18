"""Unweighted train-row standardization."""

from __future__ import annotations

from typing import Literal

from ..scaling import fit_row_standard_scaler
from .base import CompiledInputNormalizationContract, InputNormalizationConfig


class RowStandardConfig(InputNormalizationConfig[Literal["row_standard"]]):
    id: Literal["row_standard"] = "row_standard"


def compile_input_normalization(config: RowStandardConfig) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="row_standard",
        fit_scaler_fn=fit_row_standard_scaler,
    )

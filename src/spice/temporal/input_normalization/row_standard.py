"""Unweighted train-row standardization."""

from __future__ import annotations

from typing import Literal

from ..scaling import fit_row_standard_scaler
from .base import CompiledInputNormalizationContract, InputNormalizationConfig, InputNormalizationSpec
from .registry import register_input_normalization_spec


class RowStandardConfig(InputNormalizationConfig):
    id: Literal["row_standard"] = "row_standard"


def _compile(config: RowStandardConfig) -> CompiledInputNormalizationContract:
    del config
    return CompiledInputNormalizationContract(
        input_normalization_id="row_standard",
        fit_scaler_fn=fit_row_standard_scaler,
    )


register_input_normalization_spec(
    InputNormalizationSpec[RowStandardConfig](
        id="row_standard",
        config_type=RowStandardConfig,
        compile=_compile,
    )
)

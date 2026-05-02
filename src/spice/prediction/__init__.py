"""Prediction-family registry and compiled contracts."""

from .base import (
    PredictionHeadSpec,
    PredictionOutputSpec,
)
from .contracts import (
    CompiledPredictionContract,
    EpochMetricAccumulator,
    ModelInputBatch,
    PredictionBatch,
    PredictionTargetBatch,
)
from .registry import (
    compile_prediction_contract,
    validate_prediction_family_id,
)

__all__ = [
    "CompiledPredictionContract",
    "EpochMetricAccumulator",
    "ModelInputBatch",
    "PredictionBatch",
    "PredictionHeadSpec",
    "PredictionOutputSpec",
    "PredictionTargetBatch",
    "compile_prediction_contract",
    "validate_prediction_family_id",
]

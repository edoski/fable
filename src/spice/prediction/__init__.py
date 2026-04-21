"""Prediction-family registry and compiled contracts."""

from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionHeadSpec,
    PredictionOutputSpec,
    WindowMetricSummary,
)
from .contracts import (
    ActionSpaceDecodeContext,
    CompiledPredictionContract,
    DecodedOffsets,
    EpochMetricAccumulator,
    ModelInputBatch,
    PredictionBatch,
    PredictionTargetBatch,
    decode_context_from_batch,
)
from .registry import (
    compile_prediction_contract,
    validate_prediction_family_id,
)

__all__ = [
    "CompiledPredictionContract",
    "DecodedOffsets",
    "EpochMetricAccumulator",
    "ActionSpaceDecodeContext",
    "MetricDescriptor",
    "MetricSet",
    "ModelInputBatch",
    "PredictionBatch",
    "PredictionHeadSpec",
    "PredictionOutputSpec",
    "PredictionTargetBatch",
    "WindowMetricSummary",
    "compile_prediction_contract",
    "decode_context_from_batch",
    "validate_prediction_family_id",
]

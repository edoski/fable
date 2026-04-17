"""Prediction-family registry and compiled contracts."""

from .base import (
    MetricDescriptor,
    MetricSet,
    PredictionFamilyConfig,
    PredictionHeadSpec,
    PredictionOutputSpec,
    WindowMetricSummary,
)
from .contracts import (
    CompiledPredictionContract,
    EpochMetricAccumulator,
    ModelInputBatch,
    PredictionBatch,
    PredictionTargetBatch,
    bind_prediction_representation,
)
from .registry import (
    PredictionFamilySpec,
    coerce_prediction_family_config,
    compile_prediction_contract,
    prediction_family_spec,
)

__all__ = [
    "CompiledPredictionContract",
    "EpochMetricAccumulator",
    "MetricDescriptor",
    "MetricSet",
    "ModelInputBatch",
    "PredictionBatch",
    "PredictionFamilyConfig",
    "PredictionFamilySpec",
    "PredictionHeadSpec",
    "PredictionOutputSpec",
    "PredictionTargetBatch",
    "WindowMetricSummary",
    "bind_prediction_representation",
    "coerce_prediction_family_config",
    "compile_prediction_contract",
    "prediction_family_spec",
]

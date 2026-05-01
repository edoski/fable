"""Evaluation scoring runtime planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._runtime import build_cuda_modeling_runtime
from .families.base import ModelConfig
from .families.registry import resolve_model_training_precision
from .representations import RepresentationRuntimeContext


@dataclass(frozen=True, slots=True)
class EvaluationScoringRuntimePlan:
    resolved_device: torch.device
    precision: str
    representation_runtime_context: RepresentationRuntimeContext
    deterministic: bool | None
    seed: int


def build_evaluation_scoring_runtime_plan(
    *,
    model_config: ModelConfig[Any],
    batch_size: int,
    deterministic: bool | None = None,
    seed: int = 0,
) -> EvaluationScoringRuntimePlan:
    runtime = build_cuda_modeling_runtime(batch_size=batch_size)
    return EvaluationScoringRuntimePlan(
        resolved_device=runtime.resolved_device,
        precision=resolve_model_training_precision(
            device=runtime.resolved_device,
            model_config=model_config,
        ),
        representation_runtime_context=runtime.representation_runtime_context,
        deterministic=deterministic,
        seed=seed,
    )

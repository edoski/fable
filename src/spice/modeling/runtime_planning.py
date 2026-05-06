"""Modeling runtime planning."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import cast

import torch

from ..config.models import TrainingConfig
from ._runtime import (
    build_cuda_modeling_runtime,
    configure_torch_backends,
    set_global_seed,
)
from .families.base import ModelConfig
from .models import TemporalModel
from .representations import RepresentationRuntimeContext


@dataclass(frozen=True, slots=True)
class ModelingRuntimePlan:
    resolved_device: torch.device
    precision: str
    representation_runtime_context: RepresentationRuntimeContext
    deterministic: bool | None
    seed: int
    compile_enabled: bool = False


def with_representation_runtime_context(
    plan: ModelingRuntimePlan,
    runtime_context: RepresentationRuntimeContext,
) -> ModelingRuntimePlan:
    return ModelingRuntimePlan(
        resolved_device=plan.resolved_device,
        precision=plan.precision,
        representation_runtime_context=runtime_context,
        deterministic=plan.deterministic,
        seed=plan.seed,
        compile_enabled=plan.compile_enabled,
    )


def build_cuda_modeling_runtime_plan(
    *,
    model_config: ModelConfig[str],
    batch_size: int,
    deterministic: bool | None = None,
    seed: int = 0,
) -> ModelingRuntimePlan:
    del model_config
    runtime = build_cuda_modeling_runtime(batch_size=batch_size)
    return ModelingRuntimePlan(
        resolved_device=runtime.resolved_device,
        precision="32-true",
        representation_runtime_context=runtime.representation_runtime_context,
        deterministic=deterministic,
        seed=seed,
        compile_enabled=False,
    )


def build_training_modeling_runtime_plan(
    *,
    model_config: ModelConfig[str],
    training_config: TrainingConfig,
) -> ModelingRuntimePlan:
    set_global_seed(training_config.seed)
    return build_cuda_modeling_runtime_plan(
        model_config=model_config,
        batch_size=training_config.batch_size,
        deterministic=training_config.deterministic,
        seed=training_config.seed,
    )


def prepare_model_for_runtime(
    model: TemporalModel,
    plan: ModelingRuntimePlan,
) -> TemporalModel:
    model.to(plan.resolved_device)
    if plan.compile_enabled:
        return cast(TemporalModel, torch.compile(model))
    return model


def modeling_backend_scope(plan: ModelingRuntimePlan) -> AbstractContextManager[None]:
    return configure_torch_backends(
        resolved_device=plan.resolved_device,
        deterministic=plan.deterministic,
    )

"""Modeling runtime planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch_plan import BatchRuntimeContext


@dataclass(frozen=True, slots=True)
class ModelingRuntimePlan:
    resolved_device: torch.device
    precision: str
    batch_runtime_context: BatchRuntimeContext
    deterministic: bool | None
    seed: int


def with_batch_runtime_context(
    plan: ModelingRuntimePlan,
    runtime_context: BatchRuntimeContext,
) -> ModelingRuntimePlan:
    return ModelingRuntimePlan(
        resolved_device=plan.resolved_device,
        precision=plan.precision,
        batch_runtime_context=runtime_context,
        deterministic=plan.deterministic,
        seed=plan.seed,
    )


def build_cpu_modeling_runtime_plan(
    *,
    batch_size: int,
    deterministic: bool | None = None,
    seed: int = 0,
) -> ModelingRuntimePlan:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return ModelingRuntimePlan(
        resolved_device=torch.device("cpu"),
        precision="32-true",
        batch_runtime_context=BatchRuntimeContext(
            batch_size=batch_size,
        ),
        deterministic=deterministic,
        seed=seed,
    )

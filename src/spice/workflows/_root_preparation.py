"""Workflow-facing root preparation calls."""

from __future__ import annotations

from ..config.models import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig
from ..storage.workflow_root_materialization import (
    materialize_acquire_roots,
    materialize_evaluate_roots,
    materialize_train_roots,
    materialize_tune_roots,
)
from ..storage.workflow_roots import (
    AcquireWorkflowRoots,
    EvaluateWorkflowRoots,
    TrainWorkflowRoots,
    TuneWorkflowRoots,
)


def prepare_acquire_roots(config: AcquireConfig) -> AcquireWorkflowRoots:
    return materialize_acquire_roots(config)


def prepare_train_roots(config: TrainConfig) -> TrainWorkflowRoots:
    return materialize_train_roots(config)


def prepare_tune_roots(config: TuneConfig) -> TuneWorkflowRoots:
    return materialize_tune_roots(config)


def prepare_evaluate_roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    return materialize_evaluate_roots(config)

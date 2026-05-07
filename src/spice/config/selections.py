"""Unresolved workflow selections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pydantic import Field

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from .models import ProblemSpec, WorkflowTask


class WorkflowSelectionBase(ConfigModel):
    surface: str | None = None
    chain: str | None = None
    problem: str | ProblemSpec | None = None
    features: str | None = None
    storage_root: Path | None = None


class AcquireWorkflowSelection(WorkflowSelectionBase):
    provider: str | None = None
    dry_run: bool | None = None


class ModelWorkflowSelectionBase(WorkflowSelectionBase):
    objective: str | None = None
    evaluation: str | None = None
    model: str | None = None
    tuning_space: str | None = None
    training: str | None = None
    split: str | None = None
    tuning: str | None = None
    study: str | None = None


class TrainWorkflowSelection(ModelWorkflowSelectionBase):
    dataset_id: str | None = None
    study_id: str | None = None
    variant: str | None = None


class TuneWorkflowSelection(ModelWorkflowSelectionBase):
    dataset_id: str | None = None
    trial_count: int | None = Field(default=None, gt=0)


class EvaluateWorkflowSelection(ConfigModel):
    storage_root: Path | None = None
    artifact_id: str | None = None
    dataset_id: str | None = None
    evaluation: str | None = None
    delay_seconds: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)


WorkflowSelection: TypeAlias = (
    AcquireWorkflowSelection
    | TrainWorkflowSelection
    | TuneWorkflowSelection
    | EvaluateWorkflowSelection
)
SurfaceWorkflowSelection: TypeAlias = (
    AcquireWorkflowSelection | TrainWorkflowSelection | TuneWorkflowSelection
)


@dataclass(frozen=True, slots=True)
class WorkflowSelectionSpec:
    workflow: WorkflowTask
    selection_type: type[WorkflowSelection]
    benchmark_supported: bool = False


_BENCHMARK_DIMENSION_FIELDS = {
    "data": frozenset({"surface", "chain", "dataset_id"}),
    "features": frozenset({"features", "surface"}),
    "models": frozenset({"model", "tuning_space"}),
    "scoring": frozenset({"objective", "evaluation"}),
    "runtime": frozenset(
        {
            "dataset_id",
            "training",
            "split",
            "tuning",
            "study",
            "study_id",
            "artifact_id",
            "trial_count",
            "variant",
            "delay_seconds",
            "batch_size",
        }
    ),
}
_BENCHMARK_SELECTION_ROOT_FIELDS = frozenset({"dataset_id", "study_id", "artifact_id"})

_WORKFLOW_SELECTION_SPECS = (
    WorkflowSelectionSpec(WorkflowTask.ACQUIRE, AcquireWorkflowSelection),
    WorkflowSelectionSpec(WorkflowTask.TRAIN, TrainWorkflowSelection, benchmark_supported=True),
    WorkflowSelectionSpec(WorkflowTask.TUNE, TuneWorkflowSelection, benchmark_supported=True),
    WorkflowSelectionSpec(
        WorkflowTask.EVALUATE,
        EvaluateWorkflowSelection,
        benchmark_supported=True,
    ),
)
_WORKFLOW_SELECTION_SPEC_BY_TASK = {
    spec.workflow: spec for spec in _WORKFLOW_SELECTION_SPECS
}


def workflow_selection_spec(workflow: WorkflowTask) -> WorkflowSelectionSpec:
    try:
        return _WORKFLOW_SELECTION_SPEC_BY_TASK[workflow]
    except KeyError as exc:
        raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}") from exc


def workflow_selection_type(workflow: WorkflowTask) -> type[WorkflowSelection]:
    return workflow_selection_spec(workflow).selection_type


def workflow_selection_fields(workflow: WorkflowTask) -> tuple[str, ...]:
    return tuple(workflow_selection_type(workflow).model_fields)


def workflow_selection_field_set(workflow: WorkflowTask) -> frozenset[str]:
    return frozenset(workflow_selection_type(workflow).model_fields)


def benchmark_workflows() -> frozenset[WorkflowTask]:
    return frozenset(
        spec.workflow for spec in _WORKFLOW_SELECTION_SPECS if spec.benchmark_supported
    )


def benchmark_base_fields() -> frozenset[str]:
    return frozenset(
        field
        for workflow in benchmark_workflows()
        for field in workflow_selection_field_set(workflow)
    )


def benchmark_dimension_field_names() -> frozenset[str]:
    return frozenset(_BENCHMARK_DIMENSION_FIELDS)


def benchmark_dimension_fields(name: str) -> frozenset[str] | None:
    return _BENCHMARK_DIMENSION_FIELDS.get(name)


def benchmark_selection_root_fields() -> frozenset[str]:
    return _BENCHMARK_SELECTION_ROOT_FIELDS


def benchmark_selection_coordinate_fields() -> frozenset[str]:
    return benchmark_base_fields() - benchmark_selection_root_fields()

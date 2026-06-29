"""Unresolved workflow selections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pydantic import Field

from ..core.config_model import ConfigModel
from ..core.errors import ConfigResolutionError
from .models import ProblemSpec, RuntimeEvaluationWindowSpec, WorkflowTask


class WorkflowSelectionBase(ConfigModel):
    surface: str | None = None
    chain: str | None = None
    problem: str | ProblemSpec | None = None
    features: str | None = None
    storage_root: Path | None = None


class AcquireWorkflowSelection(WorkflowSelectionBase):
    corpus: str | None = None
    provider: str | None = None
    dry_run: bool | None = None


class ModelWorkflowSelectionBase(WorkflowSelectionBase):
    evaluations: str | None = None
    model: str | None = None
    tuning_space: str | None = None
    training: str | None = None
    split: str | None = None
    tuning: str | None = None
    study: str | None = None


class TrainWorkflowSelection(ModelWorkflowSelectionBase):
    corpus_id: str | None = None
    study_id: str | None = None
    variant: str | None = None


class TuneWorkflowSelection(ModelWorkflowSelectionBase):
    corpus_id: str | None = None
    trial_count: int | None = Field(default=None, gt=0)


class EvaluateWorkflowSelection(ConfigModel):
    storage_root: Path | None = None
    artifact_id: str | None = None
    corpus_id: str | None = None
    evaluation_window: RuntimeEvaluationWindowSpec | None = None
    evaluations: str | None = None
    evaluator: str | None = None
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


_WORKFLOW_SELECTION_SPECS = (
    WorkflowSelectionSpec(WorkflowTask.ACQUIRE, AcquireWorkflowSelection),
    WorkflowSelectionSpec(WorkflowTask.TRAIN, TrainWorkflowSelection),
    WorkflowSelectionSpec(WorkflowTask.TUNE, TuneWorkflowSelection),
    WorkflowSelectionSpec(WorkflowTask.EVALUATE, EvaluateWorkflowSelection),
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

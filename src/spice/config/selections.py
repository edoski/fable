"""Unresolved workflow selections."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, overload

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


def workflow_selection_type(workflow: WorkflowTask) -> type[WorkflowSelection]:
    if workflow is WorkflowTask.ACQUIRE:
        return AcquireWorkflowSelection
    if workflow is WorkflowTask.TRAIN:
        return TrainWorkflowSelection
    if workflow is WorkflowTask.TUNE:
        return TuneWorkflowSelection
    if workflow is WorkflowTask.EVALUATE:
        return EvaluateWorkflowSelection
    raise ConfigResolutionError(f"Unsupported workflow: {workflow.value}")


@overload
def workflow_selection_from_values(
    workflow: Literal[WorkflowTask.ACQUIRE],
    values: Mapping[str, object | None],
) -> AcquireWorkflowSelection: ...


@overload
def workflow_selection_from_values(
    workflow: Literal[WorkflowTask.TRAIN],
    values: Mapping[str, object | None],
) -> TrainWorkflowSelection: ...


@overload
def workflow_selection_from_values(
    workflow: Literal[WorkflowTask.TUNE],
    values: Mapping[str, object | None],
) -> TuneWorkflowSelection: ...


@overload
def workflow_selection_from_values(
    workflow: Literal[WorkflowTask.EVALUATE],
    values: Mapping[str, object | None],
) -> EvaluateWorkflowSelection: ...


@overload
def workflow_selection_from_values(
    workflow: WorkflowTask,
    values: Mapping[str, object | None],
) -> WorkflowSelection: ...


def workflow_selection_from_values(
    workflow: WorkflowTask,
    values: Mapping[str, object | None],
) -> WorkflowSelection:
    return workflow_selection_type(workflow).model_validate(
        workflow_selection_payload(workflow, values)
    )


def workflow_selection_fields(workflow: WorkflowTask) -> tuple[str, ...]:
    return tuple(workflow_selection_type(workflow).model_fields)


def workflow_selection_payload(
    workflow: WorkflowTask,
    values: Mapping[str, object | None],
) -> dict[str, object]:
    fields = frozenset(workflow_selection_fields(workflow))
    return {
        key: value
        for key, value in values.items()
        if key in fields and value is not None
    }

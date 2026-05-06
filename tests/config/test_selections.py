from __future__ import annotations

import pytest

from spice.config import (
    AcquireWorkflowSelection,
    EvaluateWorkflowSelection,
    TrainWorkflowSelection,
    TuneWorkflowSelection,
    WorkflowTask,
    coerce_problem_spec,
)
from spice.config.groups import load_named_group_payload
from spice.config.selections import (
    workflow_selection_fields,
    workflow_selection_type,
)


def test_workflow_selection_type_maps_each_workflow() -> None:
    assert workflow_selection_type(WorkflowTask.ACQUIRE) is AcquireWorkflowSelection
    assert workflow_selection_type(WorkflowTask.TRAIN) is TrainWorkflowSelection
    assert workflow_selection_type(WorkflowTask.TUNE) is TuneWorkflowSelection
    assert workflow_selection_type(WorkflowTask.EVALUATE) is EvaluateWorkflowSelection


def test_workflow_selections_validate_command_values() -> None:
    with pytest.raises(ValueError):
        TuneWorkflowSelection(
            surface="current_row_fee_dynamics",
            trial_count=0,
        )


def test_evaluate_selection_rejects_surface_fields() -> None:
    with pytest.raises(ValueError, match="surface"):
        EvaluateWorkflowSelection.model_validate(
            {
                "surface": "current_row_fee_dynamics",
                "artifact_id": "art_123",
                "dataset_id": "cor_123",
                "evaluation": "poisson_replay",
            }
        )


def test_inline_problem_spec_is_valid_for_surface_workflow_selections() -> None:
    problem = coerce_problem_spec(load_named_group_payload("current_row_nominal", "problem"))

    for workflow in (WorkflowTask.ACQUIRE, WorkflowTask.TRAIN, WorkflowTask.TUNE):
        selection_type = workflow_selection_type(workflow)
        selection = selection_type.model_validate(
            {
                "surface": "current_row_fee_dynamics",
                "problem": problem,
            }
        )

        assert "problem" in workflow_selection_fields(workflow)
        assert selection.problem == problem

    assert "problem" not in workflow_selection_fields(WorkflowTask.EVALUATE)

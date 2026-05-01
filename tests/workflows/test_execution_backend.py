from __future__ import annotations

import pytest

from spice.config import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowTask,
    resolve_workflow_config,
    workflow_config_snapshot_json,
)
from spice.config.selections import workflow_selection_type
from spice.execution.remote_runner import workflow_config_from_json

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


@pytest.mark.parametrize(
    ("task", "expected_type"),
    [
        (WorkflowTask.TRAIN, TrainConfig),
        (WorkflowTask.TUNE, TuneConfig),
        (WorkflowTask.EVALUATE, EvaluateConfig),
    ],
)
def test_remote_runner_rehydrates_resolved_workflow_snapshots(
    task: WorkflowTask,
    expected_type: type[object],
) -> None:
    selection_payload = {
        WorkflowTask.TRAIN: {
            "surface": "current_row_fee_dynamics",
            "dataset_id": TEST_DATASET_ID,
        },
        WorkflowTask.TUNE: {
            "surface": "current_row_fee_dynamics",
            "dataset_id": TEST_DATASET_ID,
        },
        WorkflowTask.EVALUATE: {
            "artifact_id": "art_test",
            "dataset_id": TEST_DATASET_ID,
            "evaluation": "poisson_replay_2h",
        },
    }[task]
    config = resolve_workflow_config(
        task,
        workflow_selection_type(task).model_validate(selection_payload),
    )

    restored = workflow_config_from_json(
        task,
        workflow_config_snapshot_json(config),
    )

    assert isinstance(restored, expected_type)
    assert restored.model_dump(mode="json") == config.model_dump(mode="json")

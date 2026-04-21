from __future__ import annotations

from typing import cast

from spice.config import TuneConfig, WorkflowTask
from spice.storage.layout import resolve_workflow_paths
from spice.storage.study_optuna import open_tuning_study


def test_tuning_objective_controls_study_direction(
    tmp_path,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    override = model_workflow_override() | tune_override()
    override["tuning"] = {
        "trial_count": 2,
        "timeout_seconds": None,
        "sampler_seed": 2026,
        "enable_pruning": False,
    }
    override["objective"] = {
        "id": "validation",
        "metric_id": "offset_accuracy",
        "direction": "maximize",
    }
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )

    access = open_tuning_study(resolve_workflow_paths(config).study_state_db, config=config)

    assert access.study.direction.name == "MAXIMIZE"
    assert access.manifest.objective.metric_id == "offset_accuracy"

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from spice.config import (
    EvaluateConfig,
    TrainConfig,
    TuneConfig,
    WorkflowTask,
    hydrate_workflow_config_snapshot,
    hydrate_workflow_config_snapshot_json,
    resolve_workflow_config,
    workflow_config_snapshot_json,
    workflow_config_snapshot_payload,
)
from spice.config.selections import workflow_selection_type
from spice.core.errors import ConfigResolutionError

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _resolved_config(task: WorkflowTask) -> TrainConfig | TuneConfig | EvaluateConfig:
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
    assert isinstance(config, (TrainConfig, TuneConfig, EvaluateConfig))
    return config


@pytest.mark.parametrize(
    ("task", "expected_type"),
    [
        (WorkflowTask.TRAIN, TrainConfig),
        (WorkflowTask.TUNE, TuneConfig),
        (WorkflowTask.EVALUATE, EvaluateConfig),
    ],
)
def test_workflow_snapshot_payload_and_json_round_trip(
    task: WorkflowTask,
    expected_type: type[object],
) -> None:
    config = _resolved_config(task)

    payload_restored = hydrate_workflow_config_snapshot(
        task,
        workflow_config_snapshot_payload(config),
    )
    json_restored = hydrate_workflow_config_snapshot_json(
        task,
        workflow_config_snapshot_json(config),
    )

    assert isinstance(payload_restored, expected_type)
    assert isinstance(json_restored, expected_type)
    assert payload_restored.model_dump(mode="json") == config.model_dump(mode="json")
    assert json_restored.model_dump(mode="json") == config.model_dump(mode="json")


def test_workflow_snapshot_preserves_owner_config_types() -> None:
    train = cast(TrainConfig, _resolved_config(WorkflowTask.TRAIN))
    tune = cast(TuneConfig, _resolved_config(WorkflowTask.TUNE))
    evaluate = cast(EvaluateConfig, _resolved_config(WorkflowTask.EVALUATE))

    restored_train = cast(
        TrainConfig,
        hydrate_workflow_config_snapshot(
            WorkflowTask.TRAIN,
            workflow_config_snapshot_payload(train),
        ),
    )
    restored_tune = cast(
        TuneConfig,
        hydrate_workflow_config_snapshot(
            WorkflowTask.TUNE,
            workflow_config_snapshot_payload(tune),
        ),
    )
    restored_evaluate = cast(
        EvaluateConfig,
        hydrate_workflow_config_snapshot(
            WorkflowTask.EVALUATE,
            workflow_config_snapshot_payload(evaluate),
        ),
    )

    assert type(restored_train.model) is type(train.model)
    assert type(restored_train.dataset_builder) is type(train.dataset_builder)
    assert type(restored_train.problem.compiler) is type(train.problem.compiler)
    assert type(restored_train.objective) is type(train.objective)
    assert type(restored_tune.tuning_space.model) is type(tune.tuning_space.model)
    assert type(restored_evaluate.evaluation) is type(evaluate.evaluation)


def test_workflow_snapshot_rejects_acquire_and_mismatched_workflow() -> None:
    train = _resolved_config(WorkflowTask.TRAIN)
    payload = workflow_config_snapshot_payload(train)

    with pytest.raises(ConfigResolutionError, match="Unsupported resolved workflow: acquire"):
        hydrate_workflow_config_snapshot(WorkflowTask.ACQUIRE, payload)

    with pytest.raises(ConfigResolutionError, match="workflow mismatch"):
        hydrate_workflow_config_snapshot(WorkflowTask.EVALUATE, payload)


def test_workflow_snapshot_reports_malformed_payloads() -> None:
    train = _resolved_config(WorkflowTask.TRAIN)
    payload = workflow_config_snapshot_payload(train)
    payload["storage"] = "not-a-mapping"

    with pytest.raises(ConfigResolutionError, match="field storage must be a mapping"):
        hydrate_workflow_config_snapshot(WorkflowTask.TRAIN, payload)

    with pytest.raises(ConfigResolutionError, match="resolved workflow snapshot must be a mapping"):
        hydrate_workflow_config_snapshot_json(WorkflowTask.TRAIN, "[]")

    with pytest.raises(ConfigResolutionError):
        hydrate_workflow_config_snapshot_json(WorkflowTask.TRAIN, "{")


def test_workflow_snapshot_excludes_none_and_supports_storage_root_override() -> None:
    evaluate = cast(EvaluateConfig, _resolved_config(WorkflowTask.EVALUATE))

    payload = workflow_config_snapshot_payload(
        evaluate,
        storage_root_override=Path("/remote/storage"),
    )

    storage_payload = cast(dict[str, object], payload["storage"])
    assert "delay_seconds" not in payload
    assert storage_payload["root"] == "/remote/storage"
    restored = cast(
        EvaluateConfig,
        hydrate_workflow_config_snapshot(WorkflowTask.EVALUATE, payload),
    )
    assert restored.storage.root == Path("/remote/storage")
    assert restored.artifact_id == evaluate.artifact_id

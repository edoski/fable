from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

import kairos.cli as cli
from kairos.cli import app
from kairos.config import EvaluateRequest, ExperimentSemantics, SelectedStudySource, TrainRequest
from tests.helpers import dispatch, window

CORPUS_ID = UUID("10000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
EVALUATION_ID = UUID("30000000-0000-4000-8000-000000000001")
STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=window(100),
        validation_window=window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas",),
    )


def _train_request() -> TrainRequest:
    source = SelectedStudySource(
        corpus_id=CORPUS_ID, study_id=STUDY_ID, study_result_index=2, experiment=_experiment()
    )
    return TrainRequest(workflow="train", artifact_id=ARTIFACT_ID, source=source)


def _evaluate_request() -> EvaluateRequest:
    return EvaluateRequest(
        workflow="evaluate",
        evaluation_id=EVALUATION_ID,
        artifact_id=ARTIFACT_ID,
        corpus_id=CORPUS_ID,
        testing_window=window(300),
    )


def test_remote_workflow_dispatches_train_and_evaluate_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_request = _train_request()
    evaluate_request = _evaluate_request()
    train_calls: list[tuple[TrainRequest, Path]] = []
    evaluate_calls: list[tuple[EvaluateRequest, Path]] = []
    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(
        cli,
        "train",
        lambda active_request, storage_root: train_calls.append((active_request, storage_root)),
    )
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda active_request, storage_root: evaluate_calls.append((active_request, storage_root)),
    )

    train_result = dispatch(app, "remote", "workflow", input=train_request.model_dump_json())
    evaluate_result = dispatch(app, "remote", "workflow", input=evaluate_request.model_dump_json())

    assert train_result.exit_code == 0
    assert train_result.output == ""
    assert evaluate_result.exit_code == 0
    assert evaluate_result.output == ""
    assert train_calls == [(train_request, STORAGE_ROOT)]
    assert evaluate_calls == [(evaluate_request, STORAGE_ROOT)]

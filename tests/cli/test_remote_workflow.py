from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest

import fable.cli as cli
from fable.cli import app
from fable.config import (
    EvaluateRequest,
    ExperimentSemantics,
    SelectedStudySource,
    TrainRequest,
    WorkflowRequest,
)
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


def _request(kind: Literal["selected", "evaluate"]) -> WorkflowRequest:
    if kind == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            testing_window=window(300),
        )
    source = SelectedStudySource(
        kind="selected_study",
        corpus_id=CORPUS_ID,
        study_id=STUDY_ID,
        study_result_index=2,
        experiment=_experiment(),
    )
    return TrainRequest(workflow="train", artifact_id=ARTIFACT_ID, source=source)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("selected", id="selected"),
        pytest.param("evaluate", id="evaluate"),
    ],
)
def test_remote_workflow_dispatches_final_request(
    kind: Literal["selected", "evaluate"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(kind)
    prepared = object()
    train_calls: list[tuple[TrainRequest, object, Path]] = []
    evaluate_calls: list[tuple[EvaluateRequest, Path]] = []
    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(cli, "load_corpus", lambda *_: object())
    monkeypatch.setattr(cli, "prepare_fit_history", lambda *_: prepared)
    monkeypatch.setattr(
        cli,
        "train",
        lambda active_request, active_prepared, storage_root: train_calls.append(
            (active_request, active_prepared, storage_root)
        ),
    )
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda active_request, storage_root: evaluate_calls.append((active_request, storage_root)),
    )

    result = dispatch(
        app,
        "remote",
        "workflow",
        input=request.model_dump_json(),
    )

    assert result.exit_code == 0
    assert result.output == ""
    if isinstance(request, TrainRequest):
        assert train_calls == [(request, prepared, STORAGE_ROOT)]
        assert evaluate_calls == []
    else:
        assert train_calls == []
        assert evaluate_calls == [(request, STORAGE_ROOT)]

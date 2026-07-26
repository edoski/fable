from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from typer.testing import CliRunner

import fable.cli.commands.remote as remote
from fable.cli.app import app
from fable.config import (
    BaselineSource,
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
    WorkflowRequest,
)

CORPUS_ID = UUID("10000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
EVALUATION_ID = UUID("30000000-0000-4000-8000-000000000001")
STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")


def _window(first: int) -> BlockWindow:
    return BlockWindow(
        first_parent_block=first,
        last_parent_block=first + 9,
    )


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=_window(100),
        validation_window=_window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas",),
    )


def _request(kind: Literal["baseline", "selected", "evaluate"]) -> WorkflowRequest:
    if kind == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            testing_window=_window(300),
        )
    if kind == "selected":
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=2,
            experiment=_experiment(),
        )
    else:
        source = BaselineSource(
            kind="baseline",
            corpus_id=CORPUS_ID,
            training_definition=TrainingDefinition(
                experiment=_experiment(),
                method=Method(
                    model=LstmDefinition(
                        family="lstm",
                        hidden=8,
                        layers=1,
                        head_hidden=4,
                        dropout=0.1,
                    ),
                    fit=FitMethod(
                        learning_rate=0.001,
                        weight_decay=0.01,
                        accumulation=1,
                        gradient_clip_norm=1.0,
                        seed=2026,
                        max_epochs=3,
                        validate_every_completed_epoch=1,
                        patience=2,
                        min_delta=0.0,
                    ),
                ),
            ),
        )
    return TrainRequest(workflow="train", artifact_id=ARTIFACT_ID, source=source)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("baseline", id="baseline"),
        pytest.param("selected", id="selected"),
        pytest.param("evaluate", id="evaluate"),
    ],
)
def test_remote_workflow_dispatches_final_request(
    kind: Literal["baseline", "selected", "evaluate"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(kind)
    prepared = object()
    train_calls: list[tuple[TrainRequest, object, Path]] = []
    evaluate_calls: list[tuple[EvaluateRequest, Path]] = []
    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(remote, "load_corpus", lambda *_: object())
    monkeypatch.setattr(remote, "prepare_fit_history", lambda *_: prepared)
    monkeypatch.setattr(
        remote,
        "train",
        lambda active_request, active_prepared, storage_root: train_calls.append(
            (active_request, active_prepared, storage_root)
        ),
    )
    monkeypatch.setattr(
        remote,
        "evaluate",
        lambda active_request, storage_root: evaluate_calls.append((active_request, storage_root)),
    )

    result = CliRunner().invoke(
        app,
        ["remote", "workflow"],
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

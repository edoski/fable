from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

import pytest
from typer.testing import CliRunner

import fable.cli.commands.remote as remote
from fable.cli.app import app
from fable.config import (
    AdamWMethod,
    BaselineSource,
    EvaluateRequest,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmDefinition,
    OriginWindow,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
    WorkflowRequest,
)
from fable.evaluation import EvaluationDeployment
from fable.modeling import FitDeployment

CORPUS_ID = UUID("10000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
EVALUATION_ID = UUID("30000000-0000-4000-8000-000000000001")
STUDY_ID = UUID("40000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")
DEPLOYMENT = {
    "evaluation_batch_size": 17,
    "num_workers": 3,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "deterministic": "warn",
    "benchmark": False,
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": False,
}


def _window(role: Literal["training", "validation", "testing"]) -> OriginWindow:
    first = {"training": 100, "validation": 210, "testing": 300}[role]
    return OriginWindow(
        role=role,
        first_parent_block=first,
        last_parent_block=first + 9,
    )


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=_window("training"),
        validation_window=_window("validation"),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas",),
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=1.0,
            classification_scale=1.0,
            regression_scale=1.0,
        ),
    )


def _request(kind: Literal["baseline", "selected", "evaluate"]) -> WorkflowRequest:
    if kind == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            window=_window("testing"),
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
                model=LstmDefinition(
                    family="lstm",
                    hidden=8,
                    layers=1,
                    head_hidden=4,
                    dropout=0.1,
                ),
                optimizer=AdamWMethod(learning_rate=0.001, weight_decay=0.01),
                training_batch=11,
                fit=FitMethod(
                    accumulation=1,
                    gradient_clip_norm=1.0,
                    scheduler="none",
                    seed=2026,
                    max_epochs=3,
                    validate_every_completed_epoch=1,
                    patience=2,
                    min_delta=0.0,
                    improvement="strict_lower",
                    restore="earliest_best",
                ),
            ),
        )
    return TrainRequest(workflow="train", artifact_id=ARTIFACT_ID, source=source)


def _payload(request: WorkflowRequest) -> bytes:
    return json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "deployment": DEPLOYMENT,
        },
        separators=(",", ":"),
    ).encode()


class _InputBuffer:
    def __init__(self, payload: bytes, events: list[str]) -> None:
        self._payload = payload
        self._events = events

    def read(self) -> bytes:
        self._events.append("stdin")
        return self._payload


class _Environment:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __getitem__(self, key: str) -> str:
        self._events.append(f"environment:{key}")
        return str(STORAGE_ROOT)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("baseline", id="baseline-train"),
        pytest.param("selected", id="selected-train"),
        pytest.param("evaluate", id="evaluate"),
    ],
)
def test_remote_workflow_executes_one_strict_envelope(
    kind: Literal["baseline", "selected", "evaluate"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(kind)
    payload = _payload(request)
    events: list[str] = []
    calls: list[tuple[str, tuple[object, ...]]] = []
    corpus = object()
    prepared = object()

    def fake_load_corpus(storage_root: Path, corpus_id: UUID) -> object:
        events.append("load_corpus")
        calls.append(("load_corpus", (storage_root, corpus_id)))
        return corpus

    def fake_prepare_fit_history(
        loaded_corpus: object,
        experiment: ExperimentSemantics,
    ) -> object:
        events.append("prepare_fit_history")
        calls.append(("prepare_fit_history", (loaded_corpus, experiment)))
        return prepared

    def fake_train(
        active_request: TrainRequest,
        active_prepared: object,
        storage_root: Path,
        deployment: FitDeployment,
    ) -> None:
        events.append("train")
        calls.append(("train", (active_request, active_prepared, storage_root, deployment)))

    def fake_evaluate(
        active_request: EvaluateRequest,
        storage_root: Path,
        deployment: EvaluationDeployment,
    ) -> None:
        events.append("evaluate")
        calls.append(("evaluate", (active_request, storage_root, deployment)))

    monkeypatch.setattr(
        remote,
        "sys",
        SimpleNamespace(stdin=SimpleNamespace(buffer=_InputBuffer(payload, events))),
    )
    monkeypatch.setattr(
        remote,
        "os",
        SimpleNamespace(environ=_Environment(events)),
    )
    monkeypatch.setattr(remote, "load_corpus", fake_load_corpus)
    monkeypatch.setattr(remote, "prepare_fit_history", fake_prepare_fit_history)
    monkeypatch.setattr(remote, "train", fake_train)
    monkeypatch.setattr(remote, "evaluate", fake_evaluate)

    result = CliRunner().invoke(app, ["remote", "workflow"])

    assert result.exit_code == 0
    assert result.return_value is None
    assert result.output == ""
    if kind == "evaluate":
        assert events == [
            "stdin",
            "environment:STORAGE_ROOT",
            "evaluate",
        ]
        assert calls == [
            (
                "evaluate",
                (
                    request,
                    STORAGE_ROOT,
                    EvaluationDeployment(
                        batch_size=17,
                        num_workers=3,
                        pin_memory=True,
                        prefetch_factor=2,
                        persistent_workers=True,
                        deterministic="warn",
                        benchmark=False,
                        float32_matmul_precision="high",
                        cuda_matmul_allow_tf32=True,
                        cudnn_allow_tf32=False,
                    ),
                ),
            )
        ]
        return

    assert isinstance(request, TrainRequest)
    expected_experiment = (
        request.source.training_definition.experiment
        if isinstance(request.source, BaselineSource)
        else request.source.experiment
    )
    expected_deployment = FitDeployment(
        deterministic="warn",
        benchmark=False,
        num_workers=3,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        float32_matmul_precision="high",
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=False,
    )
    assert events == [
        "stdin",
        "environment:STORAGE_ROOT",
        "load_corpus",
        "prepare_fit_history",
        "train",
    ]
    assert calls == [
        ("load_corpus", (STORAGE_ROOT, CORPUS_ID)),
        ("prepare_fit_history", (corpus, expected_experiment)),
        ("train", (request, prepared, STORAGE_ROOT, expected_deployment)),
    ]


def test_remote_workflow_propagates_owner_failure_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(_request("evaluate"))
    error = RuntimeError("owner failure")

    def fail_evaluate(*_args: object) -> None:
        raise error

    monkeypatch.setattr(
        remote,
        "sys",
        SimpleNamespace(stdin=SimpleNamespace(buffer=_InputBuffer(payload, []))),
    )
    monkeypatch.setattr(
        remote,
        "os",
        SimpleNamespace(environ={"STORAGE_ROOT": str(STORAGE_ROOT)}),
    )
    monkeypatch.setattr(remote, "evaluate", fail_evaluate)

    result = CliRunner().invoke(app, ["remote", "workflow"])

    assert result.exit_code == 1
    assert result.exception is error
    assert result.output == ""

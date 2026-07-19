from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import fable.cli.app as cli
from fable.cli.app import app
from fable.config import (
    WORKFLOW_REQUEST_ADAPTER,
    EvaluateRequest,
    ExperimentSemantics,
    LossDefinition,
    OriginWindow,
    SelectedStudySource,
    TrainRequest,
    WorkflowRequest,
)
from fable.execution import submit

CORPUS_ID = UUID("00000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("00000000-0000-4000-8000-000000000002")
EVALUATION_ID = UUID("00000000-0000-4000-8000-000000000003")
STUDY_ID = UUID("00000000-0000-4000-8000-000000000004")
DEPLOYMENT = {
    "evaluation_batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "deterministic": True,
    "benchmark": False,
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
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
        ordered_features=("base_fee",),
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=1.0,
            classification_scale=1.0,
            regression_scale=1.0,
        ),
    )


def _request(workflow: Literal["train", "evaluate"]) -> WorkflowRequest:
    if workflow == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            window=_window("testing"),
        )
    return TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=SelectedStudySource(
            kind="selected_study",
            corpus_id=CORPUS_ID,
            study_id=STUDY_ID,
            study_result_index=0,
            experiment=_experiment(),
        ),
    )


def _write_remote(path: Path, *, executable: str = "/opt/fable executable") -> None:
    path.write_text(
        f"""ssh: university-alias
executable: {executable}
storage_root: /remote/storage root
log_root: /remote/logs
resources:
  partition: thesis-partition
  gres: gpu:a100:1
  cpus_per_task: 8
  memory_gb: 48
  time_limit: "17:23:45"
deployment:
  evaluation_batch_size: 64
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  deterministic: true
  benchmark: false
  float32_matmul_precision: high
  cuda_matmul_allow_tf32: true
  cudnn_allow_tf32: true
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("workflow", "sbatch_output", "job_id"),
    [
        ("train", "123\n", 123),
        ("evaluate", "456;university\n", 456),
    ],
)
def test_submit_sends_one_shared_remote_profile(
    workflow: Literal["train", "evaluate"],
    sbatch_output: str,
    job_id: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(workflow)
    _write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=sbatch_output)

    monkeypatch.setattr("fable.execution.submission.subprocess.run", fake_run)

    result = submit(request)

    assert result == job_id
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "university-alias",
        "sbatch",
        "--parsable",
    ]
    envelope_json = json.dumps(
        {
            "request": json.loads(WORKFLOW_REQUEST_ADAPTER.dump_json(request)),
            "deployment": DEPLOYMENT,
        },
        separators=(",", ":"),
    )
    assert kwargs == {
        "input": (
            "#!/bin/bash\n"
            "#SBATCH --partition=thesis-partition\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --gres=gpu:a100:1\n"
            "#SBATCH --cpus-per-task=8\n"
            "#SBATCH --mem=48G\n"
            "#SBATCH --time=17:23:45\n"
            "#SBATCH --output=/remote/logs/%j.out\n"
            "export STORAGE_ROOT='/remote/storage root'\n"
            "exec '/opt/fable executable' remote workflow <<'FABLE_REQUEST'\n"
            f"{envelope_json}\n"
            "FABLE_REQUEST\n"
        ),
        "text": True,
        "stdout": subprocess.PIPE,
        "check": True,
    }


def test_submit_cli_hydrates_every_request_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_bytes(WORKFLOW_REQUEST_ADAPTER.dump_json(_request("train")))
    invalid.write_text("{}", encoding="utf-8")
    calls: list[WorkflowRequest] = []
    monkeypatch.setattr(cli, "submit_workflow", lambda request: calls.append(request) or 1)

    result = CliRunner().invoke(app, ["submit", str(valid), str(invalid)])

    assert result.exit_code == 1
    assert isinstance(result.exception, ValidationError)
    assert result.output == ""
    assert calls == []


def test_submit_cli_stops_after_failure_and_keeps_prior_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[Path] = []
    for index, workflow in enumerate(("train", "evaluate", "train")):
        path = tmp_path / f"{index}.json"
        path.write_bytes(WORKFLOW_REQUEST_ADAPTER.dump_json(_request(workflow)))
        paths.append(path)
    failure = RuntimeError("submission failed")
    calls: list[WorkflowRequest] = []

    def fake_submit(request: WorkflowRequest) -> int:
        calls.append(request)
        if len(calls) == 2:
            raise failure
        return 123

    monkeypatch.setattr(cli, "submit_workflow", fake_submit)

    result = CliRunner().invoke(app, ["submit", *(str(path) for path in paths)])

    assert result.exit_code == 1
    assert result.exception is failure
    assert result.output == "123\n"
    assert calls == [_request("train"), _request("evaluate")]


@pytest.mark.parametrize("case", ["relative_executable", "malformed_output"])
def test_submit_rejects_owned_invalid_inputs(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "REQUEST.json"
    request_path.write_bytes(WORKFLOW_REQUEST_ADAPTER.dump_json(_request("train")))
    _write_remote(
        tmp_path / "REMOTE.yaml",
        executable="relative/fable" if case == "relative_executable" else "/opt/fable",
    )
    monkeypatch.chdir(tmp_path)
    calls = 0

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, stdout="not-a-job\n")

    monkeypatch.setattr("fable.execution.submission.subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["submit", str(request_path)])

    assert result.exit_code == 1
    assert result.output == ""
    if case == "relative_executable":
        assert isinstance(result.exception, ValidationError)
        assert "executable must be an absolute path" in str(result.exception)
        assert calls == 0
    else:
        assert isinstance(result.exception, ValueError)
        assert calls == 1

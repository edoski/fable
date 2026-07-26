from __future__ import annotations

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
    BlockWindow,
    EvaluateRequest,
    ExperimentSemantics,
    SelectedStudySource,
    TrainRequest,
    WorkflowRequest,
)
from fable.execution import submit

CORPUS_ID = UUID("00000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("00000000-0000-4000-8000-000000000002")
EVALUATION_ID = UUID("00000000-0000-4000-8000-000000000003")
STUDY_ID = UUID("00000000-0000-4000-8000-000000000004")
REMOTE_YAML = """ssh: university-alias
executable: /opt/fable executable
storage_root: /remote/storage root
log_root: /remote/logs
resources:
  partition: thesis-partition
  gres: gpu:a100:1
  cpus_per_task: 8
  memory_gb: 48
  time_limit: "17:23:45"
"""


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
        ordered_features=("base_fee",),
    )


def _request(workflow: Literal["train", "evaluate"]) -> WorkflowRequest:
    if workflow == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            testing_window=_window(300),
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


def _write_remote(path: Path, contents: str = REMOTE_YAML) -> None:
    path.write_text(contents, encoding="utf-8")


def test_submit_sends_golden_workflow_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("train")
    _write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="456;university\n")

    monkeypatch.setattr("fable.execution.subprocess.run", fake_run)

    result = submit(request)

    assert result == 456
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
            f"{request.model_dump_json()}\n"
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
    valid.write_text(_request("train").model_dump_json(), encoding="utf-8")
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
    requests = [_request(workflow) for workflow in ("train", "evaluate", "train")]
    paths: list[Path] = []
    for index, request in enumerate(requests):
        path = tmp_path / f"{index}.json"
        path.write_text(request.model_dump_json(), encoding="utf-8")
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
    assert calls == requests[:2]


@pytest.mark.parametrize(
    ("remote_yaml", "message"),
    [
        (
            REMOTE_YAML.replace(
                "executable: /opt/fable executable",
                "executable: relative/fable",
            ),
            "executable must be an absolute path",
        ),
        (
            REMOTE_YAML.replace("cpus_per_task: 8", "cpus_per_task: 0"),
            "greater than 0",
        ),
    ],
)
def test_submit_rejects_raw_remote_connection_or_resource_fields(
    remote_yaml: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_remote(tmp_path / "REMOTE.yaml", remote_yaml)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match=message):
        submit(_request("train"))


def test_submit_rejects_invalid_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fable.execution.subprocess.run",
        lambda argv, **_: subprocess.CompletedProcess(
            argv,
            0,
            stdout="not-a-job\n",
        ),
    )

    with pytest.raises(ValueError, match="invalid sbatch --parsable output"):
        submit(_request("train"))

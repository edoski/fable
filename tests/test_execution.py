from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pydantic import ValidationError

import fable.cli as cli
from fable.cli import app
from fable.config import (
    EvaluateRequest,
    ExperimentSemantics,
    SelectedStudySource,
    TrainRequest,
    WorkflowRequest,
)
from fable.execution import submit_workflows
from tests.helpers import REMOTE_YAML, dispatch, window, write_remote

CORPUS_ID = UUID("00000000-0000-4000-8000-000000000001")
ARTIFACT_ID = UUID("00000000-0000-4000-8000-000000000002")
EVALUATION_ID = UUID("00000000-0000-4000-8000-000000000003")
STUDY_ID = UUID("00000000-0000-4000-8000-000000000004")


def _experiment() -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=window(100),
        validation_window=window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas",),
    )


def _request(workflow: Literal["train", "evaluate"]) -> WorkflowRequest:
    if workflow == "evaluate":
        return EvaluateRequest(
            workflow="evaluate",
            evaluation_id=EVALUATION_ID,
            artifact_id=ARTIFACT_ID,
            corpus_id=CORPUS_ID,
            testing_window=window(300),
        )
    return TrainRequest(
        workflow="train",
        artifact_id=ARTIFACT_ID,
        source=SelectedStudySource(
            corpus_id=CORPUS_ID, study_id=STUDY_ID, study_result_index=0, experiment=_experiment()
        ),
    )


def test_submit_workflows_sends_golden_single_workflow_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("train")
    write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="456;research\n")

    monkeypatch.setattr("fable.execution.subprocess.run", fake_run)

    result = submit_workflows((request,))

    assert result == 456
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["ssh", "-T", "-o", "BatchMode=yes", "research-alias", "sbatch", "--parsable"]
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
            "#SBATCH --chdir='/remote/storage root'\n"
            "export STORAGE_ROOT='/remote/storage root'\n"
            "pids=()\n"
            "srun --exclusive --exact --nodes=1 --ntasks=1 "
            "--gres=gpu:a100:1 --cpus-per-task=8 --mem=48G "
            "--output=/remote/logs/${SLURM_JOB_ID}-0.out "
            "--error=/remote/logs/${SLURM_JOB_ID}-0.out "
            "apptainer run --nv --bind '/remote/storage root' "
            "'/opt/fable image.sif' remote workflow <<'FABLE_REQUEST_0' &\n"
            f"{request.model_dump_json()}\n"
            "FABLE_REQUEST_0\n"
            'pids+=("$!")\n'
            "status=0\n"
            'for pid in "${pids[@]}"; do\n'
            '    if ! wait "$pid"; then status=1; fi\n'
            "done\n"
            'exit "$status"\n'
        ),
        "text": True,
        "stdout": subprocess.PIPE,
        "check": True,
    }


@pytest.mark.parametrize("workflow", ["train", "evaluate"])
def test_submit_cli_dispatches_request_json(
    workflow: Literal["train", "evaluate"], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(workflow)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    calls: list[tuple[WorkflowRequest, ...]] = []
    monkeypatch.setattr(
        cli, "submit_workflows", lambda submitted: calls.append(tuple(submitted)) or 123
    )

    result = dispatch(app, "submit", str(request_path))

    assert result.output == "123\n"
    assert result.exit_code == 0
    assert calls == [(request,)]


def test_submit_rejects_relative_remote_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_yaml = REMOTE_YAML.replace("image: /opt/fable image.sif", "image: relative/fable.sif")
    write_remote(tmp_path / "REMOTE.yaml", remote_yaml)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="image must be an absolute path"):
        submit_workflows((_request("train"),))


def test_submit_rejects_invalid_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fable.execution.subprocess.run",
        lambda argv, **_: subprocess.CompletedProcess(argv, 0, stdout="not-a-job\n"),
    )

    with pytest.raises(ValueError):
        submit_workflows((_request("train"),))

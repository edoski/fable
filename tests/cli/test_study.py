from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

import pytest
from typer.testing import CliRunner

import fable.cli.commands.remote as remote
import fable.cli.commands.study as study
import fable.execution.submission as submission
from fable.cli.app import app
from fable.config import (
    METHOD_ADAPTER,
    AdamWMethod,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmCapacity,
    LstmMethod,
    LstmMethodSpace,
    OriginWindow,
    StudyDefinition,
    TuneRequest,
)
from fable.modeling import FitDeployment

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")
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
FIT_DEPLOYMENT = FitDeployment(
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    deterministic=True,
    benchmark=False,
    float32_matmul_precision="high",
    cuda_matmul_allow_tf32=True,
    cudnn_allow_tf32=True,
)


def _window(role: Literal["training", "validation"]) -> OriginWindow:
    first = 100 if role == "training" else 210
    return OriginWindow(
        role=role,
        first_parent_block=first,
        last_parent_block=first + 9,
    )


METHOD = LstmMethod(
    family="lstm",
    capacity=LstmCapacity(hidden=16, layers=1, head_hidden=8),
    dropout=0.2,
    optimizer=AdamWMethod(learning_rate=3e-4, weight_decay=1e-4),
    training_batch=8,
    fit=FitMethod(
        accumulation=1,
        gradient_clip_norm=0.75,
        scheduler="none",
        seed=17,
        max_epochs=12,
        validate_every_completed_epoch=1,
        patience=4,
        min_delta=0.01,
        improvement="strict_lower",
        restore="earliest_best",
    ),
)
REQUEST = TuneRequest(
    workflow="tune",
    study_id=STUDY_ID,
    corpus_id=CORPUS_ID,
    study_definition=StudyDefinition(
        experiment=ExperimentSemantics(
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
        ),
        method_space=LstmMethodSpace(family="lstm", methods=(METHOD,)),
    ),
)


def _write_remote(path: Path) -> None:
    path.write_text(
        """ssh: university-alias
executable: /opt/fable executable
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


def test_study_run_submits_one_strict_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "TUNE_REQUEST.json"
    method_path = tmp_path / "METHOD.json"
    request_path.write_text(REQUEST.model_dump_json(), encoding="utf-8")
    method_path.write_text(METHOD.model_dump_json(), encoding="utf-8")
    _write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    calls: list[tuple[list[str], dict[str, object]]] = []
    read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        events.append(f"read:{path.name}")
        return read_bytes(path)

    class RecordingTuneRequest:
        @classmethod
        def model_validate_json(_cls, payload: bytes, *, strict: bool) -> TuneRequest:
            events.append(f"hydrate:request:{strict}")
            return TuneRequest.model_validate_json(payload, strict=strict)

    class RecordingMethodAdapter:
        def validate_json(_self, payload: bytes, *, strict: bool) -> LstmMethod:
            events.append(f"hydrate:method:{strict}")
            return METHOD_ADAPTER.validate_json(payload, strict=strict)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append("subprocess")
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="123;cluster\n")

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(study, "TuneRequest", RecordingTuneRequest)
    monkeypatch.setattr(study, "METHOD_ADAPTER", RecordingMethodAdapter())
    monkeypatch.setattr("fable.execution.submission.subprocess.run", fake_run)

    result = CliRunner().invoke(
        app,
        ["study", "run", str(request_path), str(method_path)],
    )

    assert result.exit_code == 0
    assert result.output == "123\n"
    assert events == [
        "read:TUNE_REQUEST.json",
        "hydrate:request:True",
        "read:METHOD.json",
        "hydrate:method:True",
        "read:REMOTE.yaml",
        "subprocess",
    ]
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
            "request": json.loads(REQUEST.model_dump_json()),
            "method": json.loads(METHOD.model_dump_json()),
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
            "exec '/opt/fable executable' remote candidate <<'FABLE_REQUEST'\n"
            f"{envelope_json}\n"
            "FABLE_REQUEST\n"
        ),
        "text": True,
        "stdout": subprocess.PIPE,
        "check": True,
    }


class _InputBuffer:
    def __init__(self, payload: bytes, events: list[str]) -> None:
        self._payload = payload
        self._events = events

    def read(self) -> bytes:
        self._events.append("stdin")
        return self._payload


class _Environment:
    def __init__(self, root: Path, events: list[str]) -> None:
        self._root = root
        self._events = events

    def __getitem__(self, key: str) -> str:
        self._events.append(f"environment:{key}")
        return str(self._root)


@pytest.mark.parametrize("owner_fails", [False, True])
def test_remote_candidate_forwards_one_strict_input(
    owner_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "request": json.loads(REQUEST.model_dump_json()),
            "method": json.loads(METHOD.model_dump_json()),
            "deployment": DEPLOYMENT,
        },
        separators=(",", ":"),
    ).encode()
    events: list[str] = []
    calls: list[tuple[Path, TuneRequest, LstmMethod, FitDeployment]] = []
    failure = RuntimeError("candidate failed")

    class RecordingCandidateInput:
        @classmethod
        def model_validate_json(
            _cls,
            candidate_payload: bytes,
            *,
            strict: bool,
        ):
            events.append(f"hydrate:{strict}")
            return submission._CandidateProcessInput.model_validate_json(
                candidate_payload,
                strict=strict,
            )

    def fake_run_candidate(
        storage_root: Path,
        request: TuneRequest,
        method: LstmMethod,
        deployment: FitDeployment,
    ) -> None:
        events.append("run_candidate")
        calls.append((storage_root, request, method, deployment))
        if owner_fails:
            raise failure

    monkeypatch.setattr(
        remote,
        "sys",
        SimpleNamespace(stdin=SimpleNamespace(buffer=_InputBuffer(payload, events))),
    )
    monkeypatch.setattr(
        remote,
        "os",
        SimpleNamespace(environ=_Environment(STORAGE_ROOT, events)),
    )
    monkeypatch.setattr(remote, "_CandidateProcessInput", RecordingCandidateInput)
    monkeypatch.setattr(remote, "run_candidate", fake_run_candidate)

    result = CliRunner().invoke(app, ["remote", "candidate"])

    assert result.exit_code == (1 if owner_fails else 0)
    if owner_fails:
        assert result.exception is failure
    assert result.output == ""
    assert events == [
        "stdin",
        "hydrate:True",
        "environment:STORAGE_ROOT",
        "run_candidate",
    ]
    assert calls == [(STORAGE_ROOT, REQUEST, METHOD, FIT_DEPLOYMENT)]


@pytest.mark.parametrize(
    ("study_id", "storage_root", "error"),
    [
        (STUDY_ID, Path("/current/storage"), None),
        (
            UUID("10000000-0000-1000-8000-000000000001"),
            Path("/current/storage"),
            "STUDY_ID must be a UUIDv4",
        ),
        (STUDY_ID, Path("relative/storage"), "STORAGE_ROOT must be an absolute path"),
    ],
)
def test_study_finalize_publishes_after_owned_validation(
    study_id: UUID,
    storage_root: Path,
    error: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    calls: list[tuple[Path, UUID]] = []

    def fake_publish_study(root: Path, active_study_id: UUID) -> None:
        events.append("publish_study")
        calls.append((root, active_study_id))

    monkeypatch.setattr(
        study,
        "os",
        SimpleNamespace(environ=_Environment(storage_root, events)),
    )
    monkeypatch.setattr(study, "publish_study", fake_publish_study)

    result = CliRunner().invoke(app, ["study", "finalize", str(study_id)])

    assert result.exit_code == (1 if error else 0)
    assert result.output == ""
    if error is None:
        assert events == ["environment:STORAGE_ROOT", "publish_study"]
        assert calls == [(storage_root, study_id)]
    else:
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == error
        expected_events = [] if study_id.version != 4 else ["environment:STORAGE_ROOT"]
        assert events == expected_events

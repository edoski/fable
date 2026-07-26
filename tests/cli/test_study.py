from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from typer.testing import CliRunner

import fable.cli.commands.remote as remote
import fable.cli.commands.study as study
import fable.execution as execution
from fable.cli.app import app
from fable.config import (
    BlockWindow,
    Deployment,
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TuneRequest,
)

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
DEPLOYMENT_RECORD = Deployment.model_validate(DEPLOYMENT)


def _window(first: int) -> BlockWindow:
    return BlockWindow(
        first_parent_block=first,
        last_parent_block=first + 9,
    )


METHOD = Method(
    model=LstmDefinition(
        family="lstm",
        hidden=16,
        layers=1,
        head_hidden=8,
        dropout=0.2,
    ),
    fit=FitMethod(
        learning_rate=3e-4,
        weight_decay=1e-4,
        accumulation=1,
        gradient_clip_norm=0.75,
        seed=17,
        max_epochs=12,
        validate_every_completed_epoch=1,
        patience=4,
        min_delta=0.01,
    ),
)
REQUEST = TuneRequest(
    workflow="tune",
    study_id=STUDY_ID,
    corpus_id=CORPUS_ID,
    experiment=ExperimentSemantics(
        training_window=_window(100),
        validation_window=_window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("base_fee",),
    ),
    methods=(METHOD,),
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


def test_study_run_hydrates_and_submits_one_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "TUNE_REQUEST.json"
    method_path = tmp_path / "METHOD.json"
    request_path.write_text(REQUEST.model_dump_json(), encoding="utf-8")
    method_path.write_text(METHOD.model_dump_json(), encoding="utf-8")
    _write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    scripts: list[str] = []

    def fake_invoke_sbatch(_remote: object, script: str) -> int:
        scripts.append(script)
        return 123

    monkeypatch.setattr(execution, "_invoke_sbatch", fake_invoke_sbatch)

    result = CliRunner().invoke(
        app,
        ["study", "run", str(request_path), str(method_path)],
    )

    assert result.exit_code == 0
    assert result.output == "123\n"
    envelope_json = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method": METHOD.model_dump(mode="json"),
            "deployment": DEPLOYMENT,
        },
        separators=(",", ":"),
    )
    assert scripts[0].splitlines()[-3:] == [
        "exec '/opt/fable executable' remote candidate <<'FABLE_REQUEST'",
        envelope_json,
        "FABLE_REQUEST",
    ]


class _InputBuffer:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


@pytest.mark.parametrize("owner_fails", [False, True])
def test_remote_candidate_forwards_input(
    owner_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method": METHOD.model_dump(mode="json"),
            "deployment": DEPLOYMENT,
        },
        separators=(",", ":"),
    ).encode()
    calls: list[tuple[Path, TuneRequest, Method, Deployment]] = []
    failure = RuntimeError("candidate failed")
    candidate = remote._CandidateProcessInput.model_validate_json(payload, strict=True)

    def fake_run_candidate(
        storage_root: Path,
        request: TuneRequest,
        method: Method,
        deployment: Deployment,
    ) -> None:
        calls.append((storage_root, request, method, deployment))
        if owner_fails:
            raise failure

    monkeypatch.setattr(
        remote,
        "sys",
        SimpleNamespace(stdin=SimpleNamespace(buffer=_InputBuffer(payload))),
    )
    monkeypatch.setattr(
        remote._CandidateProcessInput,
        "model_validate_json",
        staticmethod(lambda *_args, **_kwargs: candidate),
    )
    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(remote, "run_candidate", fake_run_candidate)

    result = CliRunner().invoke(app, ["remote", "candidate"])

    assert result.exit_code == (1 if owner_fails else 0)
    if owner_fails:
        assert result.exception is failure
    assert result.output == ""
    assert calls == [(STORAGE_ROOT, REQUEST, METHOD, DEPLOYMENT_RECORD)]
    assert calls[0][-1] is candidate.deployment


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
    calls: list[tuple[Path, UUID]] = []

    def fake_publish_study(root: Path, active_study_id: UUID) -> None:
        calls.append((root, active_study_id))

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setattr(study, "publish_study", fake_publish_study)

    result = CliRunner().invoke(app, ["study", "finalize", str(study_id)])

    assert result.exit_code == (1 if error else 0)
    assert result.output == ""
    if error is None:
        assert calls == [(storage_root, study_id)]
    else:
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == error
        assert calls == []

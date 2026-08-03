from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from typer.testing import CliRunner

import kairos.cli as cli
import kairos.execution as execution
from kairos.cli import app
from kairos.config import ExperimentSemantics, FitMethod, LstmDefinition, Method, TuneRequest
from kairos.execution import CandidateProcessInput
from tests.helpers import dispatch, window, write_remote

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")


METHOD = Method(
    model=LstmDefinition(family="lstm", hidden=16, layers=1, head_hidden=8, dropout=0.2),
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
        training_window=window(100),
        validation_window=window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("log_base_fee_per_gas",),
    ),
    methods=(METHOD,),
)


def test_study_run_submits_typed_candidate_and_prints_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "TUNE_REQUEST.json"
    request_path.write_text(REQUEST.model_dump_json(), encoding="utf-8")
    calls: list[tuple[CandidateProcessInput, ...]] = []
    monkeypatch.setattr(
        cli, "submit_candidates", lambda candidates: calls.append(tuple(candidates)) or 123
    )

    result = CliRunner().invoke(app, ["study", "run", str(request_path), "0"])

    assert result.exit_code == 0
    assert result.output == "123\n"
    assert calls == [(CandidateProcessInput(request=REQUEST, method_index=0),)]


def test_submit_candidates_scales_three_gpu_allocation_and_preserves_payload_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_request = REQUEST.model_copy(
        update={"study_id": UUID("10000000-0000-4000-8000-000000000002")}
    )
    third_request = REQUEST.model_copy(
        update={"study_id": UUID("10000000-0000-4000-8000-000000000003")}
    )
    candidates = (
        CandidateProcessInput(request=REQUEST, method_index=0),
        CandidateProcessInput(request=second_request, method_index=0),
        CandidateProcessInput(request=third_request, method_index=0),
    )
    write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    scripts: list[str] = []
    monkeypatch.setattr(
        execution, "_invoke_sbatch", lambda _remote, script: scripts.append(script) or 456
    )

    result = execution.submit_candidates(candidates)

    assert result == 456
    assert len(scripts) == 1
    script = scripts[0]
    assert "#SBATCH --ntasks=3\n" in script
    assert "#SBATCH --gres=gpu:a100:3\n" in script
    assert script.count("remote candidate <<'KAIROS_REQUEST_") == 3
    positions = [script.index(candidate.model_dump_json()) for candidate in candidates]
    assert positions == sorted(positions)
    for slot in range(3):
        assert f"--output=/remote/logs/${{SLURM_JOB_ID}}-{slot}.out" in script
        assert f"--error=/remote/logs/${{SLURM_JOB_ID}}-{slot}.out" in script


def test_remote_candidate_dispatches_input(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {"request": REQUEST.model_dump(mode="json"), "method_index": 0}, separators=(",", ":")
    )
    calls: list[tuple[Path, TuneRequest, int]] = []

    def fake_run_candidate(storage_root: Path, request: TuneRequest, method_index: int) -> None:
        calls.append((storage_root, request, method_index))

    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(cli, "run_candidate", fake_run_candidate)

    result = dispatch(app, "remote", "candidate", input=payload)

    assert result.exit_code == 0
    assert result.output == ""
    assert calls == [(STORAGE_ROOT, REQUEST, 0)]


def test_remote_candidate_rejects_method_index_outside_request() -> None:
    payload = json.dumps(
        {"request": REQUEST.model_dump(mode="json"), "method_index": 1}, separators=(",", ":")
    )

    result = dispatch(app, "remote", "candidate", input=payload)

    assert result.exit_code == 1
    assert "method_index must identify a request Method" in str(result.exception)

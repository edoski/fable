from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from typer.testing import CliRunner

import fable.cli as cli
import fable.execution as execution
from fable.cli import app
from fable.config import (
    ExperimentSemantics,
    FitMethod,
    LstmDefinition,
    Method,
    TuneRequest,
)
from tests.helpers import dispatch, window, write_remote

STUDY_ID = UUID("10000000-0000-4000-8000-000000000001")
CORPUS_ID = UUID("20000000-0000-4000-8000-000000000001")
STORAGE_ROOT = Path("/remote/storage root")


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
        training_window=window(100),
        validation_window=window(210),
        context_blocks=20,
        horizon_blocks=10,
        ordered_features=("base_fee",),
    ),
    methods=(METHOD,),
)


def test_study_run_sends_golden_candidate_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "TUNE_REQUEST.json"
    method_path = tmp_path / "METHOD.json"
    request_path.write_text(REQUEST.model_dump_json(), encoding="utf-8")
    method_path.write_text(METHOD.model_dump_json(), encoding="utf-8")
    write_remote(tmp_path / "REMOTE.yaml")
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
    candidate_json = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method": METHOD.model_dump(mode="json"),
        },
        separators=(",", ":"),
    )
    assert scripts == [
        (
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
            f"{candidate_json}\n"
            "FABLE_REQUEST\n"
        )
    ]


def test_remote_candidate_dispatches_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method": METHOD.model_dump(mode="json"),
        },
        separators=(",", ":"),
    )
    calls: list[tuple[Path, TuneRequest, Method]] = []

    def fake_run_candidate(
        storage_root: Path,
        request: TuneRequest,
        method: Method,
    ) -> None:
        calls.append((storage_root, request, method))

    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(cli, "run_candidate", fake_run_candidate)

    result = dispatch(app, "remote", "candidate", input=payload)

    assert result.exit_code == 0
    assert result.output == ""
    assert calls == [(STORAGE_ROOT, REQUEST, METHOD)]

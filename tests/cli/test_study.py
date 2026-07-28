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
from fable.execution import CandidateProcessInput
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
        ordered_features=("log_base_fee_per_gas",),
    ),
    methods=(METHOD,),
)


def test_study_run_sends_golden_candidate_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "TUNE_REQUEST.json"
    request_path.write_text(REQUEST.model_dump_json(), encoding="utf-8")
    write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    scripts: list[str] = []

    def fake_invoke_sbatch(_remote: object, script: str) -> int:
        scripts.append(script)
        return 123

    monkeypatch.setattr(execution, "_invoke_sbatch", fake_invoke_sbatch)

    result = CliRunner().invoke(
        app,
        ["study", "run", str(request_path), "0"],
    )

    assert result.exit_code == 0
    assert result.output == "123\n"
    candidate_json = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method_index": 0,
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
            "#SBATCH --chdir='/remote/storage root'\n"
            "export STORAGE_ROOT='/remote/storage root'\n"
            "pids=()\n"
            "srun --exclusive --exact --nodes=1 --ntasks=1 "
            "--gres=gpu:a100:1 --cpus-per-task=8 --mem=48G "
            "apptainer run --nv --bind '/remote/storage root' "
            "'/opt/fable image.sif' remote candidate <<'FABLE_REQUEST_0' &\n"
            f"{candidate_json}\n"
            "FABLE_REQUEST_0\n"
            'pids+=("$!")\n'
            "status=0\n"
            'for pid in "${pids[@]}"; do\n'
            '    if ! wait "$pid"; then status=1; fi\n'
            "done\n"
            'exit "$status"\n'
        )
    ]


def test_submit_candidate_batch_runs_each_candidate_on_one_exclusive_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        execution,
        "_invoke_sbatch",
        lambda _remote, script: scripts.append(script) or 456,
    )

    result = execution.submit_candidate_batch(candidates)

    assert result == 456
    assert scripts == [
        (
            "#!/bin/bash\n"
            "#SBATCH --partition=thesis-partition\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=3\n"
            "#SBATCH --gres=gpu:a100:3\n"
            "#SBATCH --cpus-per-task=8\n"
            "#SBATCH --mem=144G\n"
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
            "'/opt/fable image.sif' remote candidate <<'FABLE_REQUEST_0' &\n"
            f"{candidates[0].model_dump_json()}\n"
            "FABLE_REQUEST_0\n"
            'pids+=("$!")\n'
            "srun --exclusive --exact --nodes=1 --ntasks=1 "
            "--gres=gpu:a100:1 --cpus-per-task=8 --mem=48G "
            "--output=/remote/logs/${SLURM_JOB_ID}-1.out "
            "--error=/remote/logs/${SLURM_JOB_ID}-1.out "
            "apptainer run --nv --bind '/remote/storage root' "
            "'/opt/fable image.sif' remote candidate <<'FABLE_REQUEST_1' &\n"
            f"{candidates[1].model_dump_json()}\n"
            "FABLE_REQUEST_1\n"
            'pids+=("$!")\n'
            "srun --exclusive --exact --nodes=1 --ntasks=1 "
            "--gres=gpu:a100:1 --cpus-per-task=8 --mem=48G "
            "--output=/remote/logs/${SLURM_JOB_ID}-2.out "
            "--error=/remote/logs/${SLURM_JOB_ID}-2.out "
            "apptainer run --nv --bind '/remote/storage root' "
            "'/opt/fable image.sif' remote candidate <<'FABLE_REQUEST_2' &\n"
            f"{candidates[2].model_dump_json()}\n"
            "FABLE_REQUEST_2\n"
            'pids+=("$!")\n'
            "status=0\n"
            'for pid in "${pids[@]}"; do\n'
            '    if ! wait "$pid"; then status=1; fi\n'
            "done\n"
            'exit "$status"\n'
        )
    ]


def test_submit_candidate_batch_rejects_duplicate_study_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CandidateProcessInput(request=REQUEST, method_index=0)
    write_remote(tmp_path / "REMOTE.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        execution,
        "_invoke_sbatch",
        lambda *_: pytest.fail("duplicate candidates must fail before submission"),
    )

    with pytest.raises(ValueError, match="packed candidate slots must be unique"):
        execution.submit_candidate_batch((candidate, candidate, candidate))


def test_remote_candidate_dispatches_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method_index": 0,
        },
        separators=(",", ":"),
    )
    calls: list[tuple[Path, TuneRequest, int]] = []

    def fake_run_candidate(
        storage_root: Path,
        request: TuneRequest,
        method_index: int,
    ) -> None:
        calls.append((storage_root, request, method_index))

    monkeypatch.setenv("STORAGE_ROOT", str(STORAGE_ROOT))
    monkeypatch.setattr(cli, "run_candidate", fake_run_candidate)

    result = dispatch(app, "remote", "candidate", input=payload)

    assert result.exit_code == 0
    assert result.output == ""
    assert calls == [(STORAGE_ROOT, REQUEST, 0)]


def test_remote_candidate_rejects_method_index_outside_request() -> None:
    payload = json.dumps(
        {
            "request": REQUEST.model_dump(mode="json"),
            "method_index": 1,
        },
        separators=(",", ":"),
    )

    result = dispatch(app, "remote", "candidate", input=payload)

    assert result.exit_code == 1
    assert "method_index must identify a request Method" in str(result.exception)

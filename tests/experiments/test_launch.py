from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from fable.config import EvaluateRequest
from tests.helpers import dispatch, read_tsv_rows, run_script, window

_ROOT = Path(__file__).parents[2]
_FEATURE_SCRIPT = _ROOT / "experiments" / "feature_ablation.py"
_LAUNCH_SCRIPT = _ROOT / "experiments" / "launch.py"


def _load_launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(_ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location("experiment_launch", _LAUNCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidates_submit_three_per_job_and_record_exact_cell_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = UUID(run_script(_FEATURE_SCRIPT, "prepare", tmp_path).stdout.strip())
    bundle = tmp_path / "experiments" / "feature_ablation" / f".{experiment_id}"
    launcher = _load_launcher(monkeypatch)
    batches: list[tuple[object, ...]] = []

    def submit(candidates: tuple[object, ...]) -> int:
        batches.append(candidates)
        return 1_000 + len(batches)

    monkeypatch.setattr(launcher, "submit_candidates", submit)
    result = dispatch(launcher.app, "candidates", str(bundle))

    assert result.exit_code == 0
    assert result.output.splitlines() == [str(job_id) for job_id in range(1_001, 1_035)]
    assert [len(batch) for batch in batches] == [3] * 34
    jobs = read_tsv_rows(bundle / "jobs.tsv")
    assert len(jobs) == 102
    assert jobs[:3] == [
        {
            "job_id": "1001",
            "slot": "0",
            "row": "0",
            "cell": "ethereum.lstm.full",
        },
        {
            "job_id": "1001",
            "slot": "1",
            "row": "1",
            "cell": "ethereum.lstm.without_base_fee",
        },
        {
            "job_id": "1001",
            "slot": "2",
            "row": "2",
            "cell": "ethereum.lstm.without_gas_utilization",
        },
    ]

    repeated = dispatch(launcher.app, "candidates", str(bundle))

    assert repeated.exit_code == 0
    assert repeated.output == ""
    assert len(batches) == 34


def test_workflows_submit_two_per_job_and_record_exact_cell_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    requests = bundle / "requests"
    requests.mkdir(parents=True)
    rows: list[tuple[str, Path]] = []
    for index in range(6):
        request = EvaluateRequest(
            workflow="evaluate",
            evaluation_id=UUID(f"10000000-0000-4000-8000-{index + 1:012d}"),
            artifact_id=UUID(f"20000000-0000-4000-8000-{index + 1:012d}"),
            corpus_id=UUID("30000000-0000-4000-8000-000000000001"),
            testing_window=window(300),
        )
        path = requests / f"{index}.json"
        path.write_text(request.model_dump_json(), encoding="utf-8")
        rows.append((f"cell-{index}", path))
    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "request"))
        writer.writerows(rows)

    launcher = _load_launcher(monkeypatch)
    batches: list[tuple[object, ...]] = []

    def submit(request_batch: tuple[object, ...]) -> int:
        batches.append(request_batch)
        return 2_000 + len(batches)

    monkeypatch.setattr(launcher, "submit_workflows", submit)

    result = dispatch(
        launcher.app,
        "workflows",
        str(bundle),
        "--tasks-per-job",
        "2",
    )

    assert result.exit_code == 0
    assert result.output == "2001\n2002\n2003\n"
    assert [len(batch) for batch in batches] == [2, 2, 2]
    assert read_tsv_rows(bundle / "jobs.tsv") == [
        {"job_id": "2001", "slot": "0", "row": "0", "cell": "cell-0"},
        {"job_id": "2001", "slot": "1", "row": "1", "cell": "cell-1"},
        {"job_id": "2002", "slot": "0", "row": "2", "cell": "cell-2"},
        {"job_id": "2002", "slot": "1", "row": "3", "cell": "cell-3"},
        {"job_id": "2003", "slot": "0", "row": "4", "cell": "cell-4"},
        {"job_id": "2003", "slot": "1", "row": "5", "cell": "cell-5"},
    ]


def test_jobs_rejects_more_than_three_rows_for_one_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher(monkeypatch)
    rows = [{"cell": f"cell-{index}"} for index in range(4)]
    jobs_path = tmp_path / "jobs.tsv"
    with jobs_path.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("job_id", "slot", "row", "cell"))
        writer.writerows(
            ((123, slot, row, rows[row]["cell"]) for row, slot in enumerate((0, 0, 1, 2)))
        )

    with pytest.raises(ValueError, match="complete packed allocations"):
        launcher._load_submitted_rows(jobs_path, rows)

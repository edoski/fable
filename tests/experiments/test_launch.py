from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from fable.config import EvaluateRequest
from fable.execution import CandidateProcessInput
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


def _write_workflow_bundle(root: Path, count: int) -> tuple[Path, list[UUID]]:
    bundle = root / "bundle"
    requests = bundle / "requests"
    requests.mkdir(parents=True)
    rows: list[tuple[str, Path]] = []
    evaluation_ids: list[UUID] = []
    for index in range(count):
        evaluation_id = UUID(f"10000000-0000-4000-8000-{index + 1:012d}")
        request = EvaluateRequest(
            workflow="evaluate",
            evaluation_id=evaluation_id,
            artifact_id=UUID(f"20000000-0000-4000-8000-{index + 1:012d}"),
            corpus_id=UUID("30000000-0000-4000-8000-000000000001"),
            testing_window=window(300),
        )
        path = requests / f"{index}.json"
        path.write_text(request.model_dump_json(), encoding="utf-8")
        rows.append((f"cell-{index}", path))
        evaluation_ids.append(evaluation_id)
    with (bundle / "cells.tsv").open("x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "request"))
        writer.writerows(rows)
    return bundle, evaluation_ids


def test_candidates_submit_typed_inputs_and_restart_skips_recorded_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert result.output.splitlines() == [str(job_id) for job_id in range(1_001, 1_027)]
    assert [len(batch) for batch in batches] == [4] * 24 + [3, 3]
    assert all(
        isinstance(candidate, CandidateProcessInput) for batch in batches for candidate in batch
    )
    jobs = read_tsv_rows(bundle / "jobs.tsv")
    assert len(jobs) == 102
    assert jobs[:4] == [
        {"job_id": "1001", "slot": "0", "row": "0", "cell": "ethereum.lstm.full"},
        {"job_id": "1001", "slot": "1", "row": "1", "cell": "ethereum.lstm.without_base_fee"},
        {
            "job_id": "1001",
            "slot": "2",
            "row": "2",
            "cell": "ethereum.lstm.without_gas_utilization",
        },
        {
            "job_id": "1001",
            "slot": "3",
            "row": "3",
            "cell": "ethereum.lstm.without_exact_forming_base_fee",
        },
    ]

    repeated = dispatch(launcher.app, "candidates", str(bundle))

    assert repeated.exit_code == 0
    assert repeated.output == ""
    assert len(batches) == 26


@pytest.mark.parametrize(
    ("count", "capacity", "expected_sizes"),
    (
        (7, 4, [4, 3]),
        (9, 4, [3, 3, 3]),
        (45, 4, [4] * 9 + [3] * 3),
        (7, 3, [3, 2, 2]),
        (8, 3, [3, 3, 2]),
        (3, 3, [3]),
        (1, 3, [1]),
        (7, 2, [2, 2, 2, 1]),
    ),
)
def test_workflows_use_fewest_ordered_allocations_without_avoidable_singletons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    capacity: int,
    expected_sizes: list[int],
) -> None:
    bundle, evaluation_ids = _write_workflow_bundle(tmp_path, count)
    launcher = _load_launcher(monkeypatch)
    batches: list[tuple[object, ...]] = []

    def submit(request_batch: tuple[object, ...]) -> int:
        batches.append(request_batch)
        return 2_000 + len(batches)

    monkeypatch.setattr(launcher, "submit_workflows", submit)
    result = dispatch(launcher.app, "workflows", str(bundle), "--tasks-per-job", str(capacity))

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        str(job_id) for job_id in range(2_001, 2_001 + len(expected_sizes))
    ]
    assert [len(batch) for batch in batches] == expected_sizes
    assert [
        request.evaluation_id
        for batch in batches
        for request in batch
        if isinstance(request, EvaluateRequest)
    ] == evaluation_ids
    jobs = read_tsv_rows(bundle / "jobs.tsv")
    assert [int(job["row"]) for job in jobs] == list(range(count))
    assert [job["cell"] for job in jobs] == [f"cell-{index}" for index in range(count)]


def test_launch_persists_each_success_and_failure_leaves_later_groups_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, evaluation_ids = _write_workflow_bundle(tmp_path, 8)
    launcher = _load_launcher(monkeypatch)
    batches: list[tuple[EvaluateRequest, ...]] = []

    def fail_second(request_batch: tuple[EvaluateRequest, ...]) -> int:
        batches.append(request_batch)
        if len(batches) == 2:
            raise RuntimeError("submission failed")
        return 3_001

    monkeypatch.setattr(launcher, "submit_workflows", fail_second)

    failed = dispatch(launcher.app, "workflows", str(bundle))

    assert failed.exit_code == 1
    assert isinstance(failed.exception, RuntimeError)
    assert [len(batch) for batch in batches] == [4, 4]
    assert read_tsv_rows(bundle / "jobs.tsv") == [
        {"job_id": "3001", "slot": str(slot), "row": str(slot), "cell": f"cell-{slot}"}
        for slot in range(4)
    ]

    resumed_batches: list[tuple[EvaluateRequest, ...]] = []

    def resume(request_batch: tuple[EvaluateRequest, ...]) -> int:
        resumed_batches.append(request_batch)
        return 4_000 + len(resumed_batches)

    monkeypatch.setattr(launcher, "submit_workflows", resume)
    resumed = dispatch(launcher.app, "workflows", str(bundle))

    assert resumed.exit_code == 0
    assert resumed.output == "4001\n"
    assert [len(batch) for batch in resumed_batches] == [4]
    assert [
        request.evaluation_id for batch in resumed_batches for request in batch
    ] == evaluation_ids[4:]
    assert [int(job["row"]) for job in read_tsv_rows(bundle / "jobs.tsv")] == list(range(8))

    replay = dispatch(launcher.app, "workflows", str(bundle))

    assert replay.exit_code == 0
    assert replay.output == ""
    assert [len(batch) for batch in resumed_batches] == [4]

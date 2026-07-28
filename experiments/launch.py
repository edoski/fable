"""Launch one experiment bundle in packed GPU allocations."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import typer
from bundle import read_cells

from fable.config import WORKFLOW_REQUEST_ADAPTER, TuneRequest
from fable.execution import (
    MAX_PACKED_PROCESS_COUNT,
    CandidateProcessInput,
    submit_candidate_batch,
    submit_workflow_batch,
)

_ProcessInput = TypeVar("_ProcessInput")


def candidates(bundle: Path, tasks_per_job: int = MAX_PACKED_PROCESS_COUNT) -> None:
    bundle = bundle.resolve()
    rows = read_cells(bundle)
    process_inputs: list[CandidateProcessInput] = []
    for row in rows:
        request = TuneRequest.model_validate_json(
            Path(row["request"]).read_bytes(),
            strict=True,
        )
        if request.study_id != UUID(row["study_id"]):
            raise ValueError("candidate row Study ID must match its request")
        process_inputs.append(
            CandidateProcessInput(
                request=request,
                method_index=int(row["method_index"]),
            )
        )
    _launch(bundle, rows, process_inputs, submit_candidate_batch, tasks_per_job)


def workflows(bundle: Path, tasks_per_job: int = MAX_PACKED_PROCESS_COUNT) -> None:
    bundle = bundle.resolve()
    rows = read_cells(bundle)
    process_inputs = [
        WORKFLOW_REQUEST_ADAPTER.validate_json(
            Path(row["request"]).read_bytes(),
            strict=True,
        )
        for row in rows
    ]
    _launch(bundle, rows, process_inputs, submit_workflow_batch, tasks_per_job)


def _launch(
    bundle: Path,
    rows: list[dict[str, str]],
    process_inputs: Sequence[_ProcessInput],
    submit: Callable[[Sequence[_ProcessInput]], int],
    tasks_per_job: int,
) -> None:
    if not rows:
        raise ValueError("experiment bundle must contain at least one cell")
    if not 2 <= tasks_per_job <= MAX_PACKED_PROCESS_COUNT:
        raise ValueError("tasks per job must be two or three")

    jobs_path = bundle / "jobs.tsv"
    submitted_rows = _load_submitted_rows(jobs_path, rows)
    pending = [
        (index, row, process_input)
        for index, (row, process_input) in enumerate(
            zip(rows, process_inputs, strict=True)
        )
        if index not in submitted_rows
    ]
    if not pending:
        return
    if len(pending) % tasks_per_job:
        raise ValueError("pending experiment rows must fill packed allocations")

    exists = jobs_path.exists()
    with jobs_path.open("a" if exists else "x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writerow(("job_id", "slot", "row", "cell"))
        for start in range(0, len(pending), tasks_per_job):
            group = pending[start : start + tasks_per_job]
            job_id = submit(tuple(process_input for _, _, process_input in group))
            for slot, (row_index, row, _) in enumerate(group):
                writer.writerow((job_id, slot, row_index, row["cell"]))
            destination.flush()
            os.fsync(destination.fileno())
            print(job_id)


def _load_submitted_rows(
    jobs_path: Path,
    rows: list[dict[str, str]],
) -> set[int]:
    if not jobs_path.exists():
        return set()

    with jobs_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["job_id", "slot", "row", "cell"]:
            raise ValueError("jobs.tsv must have the exact packed-launch schema")
        jobs = list(reader)

    submitted_rows = {int(job["row"]) for job in jobs}
    if len(submitted_rows) != len(jobs):
        raise ValueError("jobs.tsv must identify unique experiment rows")

    slots_by_job: dict[int, list[int]] = {}
    for job in jobs:
        job_id = int(job["job_id"])
        slot = int(job["slot"])
        row_index = int(job["row"])
        if (
            job_id <= 0
            or not 0 <= slot < MAX_PACKED_PROCESS_COUNT
            or not 0 <= row_index < len(rows)
            or job["cell"] != rows[row_index]["cell"]
        ):
            raise ValueError("jobs.tsv must contain valid job IDs and slots")
        slots_by_job.setdefault(job_id, []).append(slot)
    if any(
        not 2 <= len(slots) <= MAX_PACKED_PROCESS_COUNT
        or sorted(slots) != list(range(len(slots)))
        for slots in slots_by_job.values()
    ):
        raise ValueError("jobs.tsv must contain complete packed allocations")
    return submitted_rows


app = typer.Typer(add_completion=False)
app.command()(candidates)
app.command()(workflows)


if __name__ == "__main__":
    app()

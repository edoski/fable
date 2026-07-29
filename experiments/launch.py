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
    MAX_ALLOCATION_PROCESS_COUNT,
    CandidateProcessInput,
    submit_candidates,
    submit_workflows,
)

_ProcessInput = TypeVar("_ProcessInput")


def candidates(bundle: Path, tasks_per_job: int = MAX_ALLOCATION_PROCESS_COUNT) -> None:
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
    _launch(bundle, rows, process_inputs, submit_candidates, tasks_per_job)


def workflows(bundle: Path, tasks_per_job: int = MAX_ALLOCATION_PROCESS_COUNT) -> None:
    bundle = bundle.resolve()
    rows = read_cells(bundle)
    process_inputs = [
        WORKFLOW_REQUEST_ADAPTER.validate_json(
            Path(row["request"]).read_bytes(),
            strict=True,
        )
        for row in rows
    ]
    _launch(bundle, rows, process_inputs, submit_workflows, tasks_per_job)


def _launch(
    bundle: Path,
    rows: list[dict[str, str]],
    process_inputs: Sequence[_ProcessInput],
    submit: Callable[[Sequence[_ProcessInput]], int],
    tasks_per_job: int,
) -> None:
    if not rows:
        raise ValueError("experiment bundle must contain at least one cell")
    if not 2 <= tasks_per_job <= MAX_ALLOCATION_PROCESS_COUNT:
        raise ValueError("tasks per job must be two or three")

    jobs_path = bundle / "jobs.tsv"
    submitted_rows = _load_submitted_rows(jobs_path)
    pending = [
        (index, row, process_input)
        for index, (row, process_input) in enumerate(zip(rows, process_inputs, strict=True))
        if index not in submitted_rows
    ]
    if not pending:
        return

    exists = jobs_path.exists()
    with jobs_path.open("a" if exists else "x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writerow(("job_id", "slot", "row", "cell"))
        start = 0
        for group_size in _allocation_sizes(len(pending), tasks_per_job):
            group = pending[start : start + group_size]
            start += group_size
            job_id = submit(tuple(process_input for _, _, process_input in group))
            for slot, (row_index, row, _) in enumerate(group):
                writer.writerow((job_id, slot, row_index, row["cell"]))
            destination.flush()
            os.fsync(destination.fileno())
            print(job_id)


def _allocation_sizes(pending_count: int, capacity: int) -> list[int]:
    full_allocations, remainder = divmod(pending_count, capacity)
    sizes = [capacity] * full_allocations
    if remainder == 1 and capacity == 3 and pending_count > 1:
        sizes[-1] = 2
        sizes.append(2)
    elif remainder:
        sizes.append(remainder)
    return sizes


def _load_submitted_rows(jobs_path: Path) -> set[int]:
    if not jobs_path.exists():
        return set()

    with jobs_path.open(newline="", encoding="utf-8") as source:
        return {int(job["row"]) for job in csv.DictReader(source, delimiter="\t")}


app = typer.Typer(add_completion=False)
app.command()(candidates)
app.command()(workflows)


if __name__ == "__main__":
    app()

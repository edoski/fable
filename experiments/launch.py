"""Launch one experiment bundle in packed GPU allocations."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TypeVar

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
        request = TuneRequest.model_validate_json(Path(row["request"]).read_bytes(), strict=True)
        process_inputs.append(
            CandidateProcessInput(request=request, method_index=int(row["method_index"]))
        )
    storage_root = bundle.parents[2]
    completed_rows = {
        index
        for index, row in enumerate(rows)
        if (storage_root / "studies" / f"{row['study_id']}.json").is_file()
    }
    _launch(
        bundle,
        rows,
        process_inputs,
        submit_candidates,
        tasks_per_job,
        completed_rows=completed_rows,
    )


def workflows(bundle: Path, tasks_per_job: int = MAX_ALLOCATION_PROCESS_COUNT) -> None:
    bundle = bundle.resolve()
    rows = read_cells(bundle)
    process_inputs = [
        WORKFLOW_REQUEST_ADAPTER.validate_json(Path(row["request"]).read_bytes(), strict=True)
        for row in rows
    ]
    _launch(bundle, rows, process_inputs, submit_workflows, tasks_per_job)


def _launch(
    bundle: Path,
    rows: list[dict[str, str]],
    process_inputs: Sequence[_ProcessInput],
    submit: Callable[[Sequence[_ProcessInput]], int],
    tasks_per_job: int,
    completed_rows: Collection[int] = (),
) -> None:
    if not 2 <= tasks_per_job <= MAX_ALLOCATION_PROCESS_COUNT:
        raise ValueError("tasks per job must be between two and four")

    jobs_path = bundle / "jobs.tsv"
    jobs_exist = jobs_path.exists()
    submitted_rows = _load_submitted_rows(jobs_path) if jobs_exist else set()
    pending = [
        (index, row, process_input)
        for index, (row, process_input) in enumerate(zip(rows, process_inputs, strict=True))
        if index not in submitted_rows and index not in completed_rows
    ]
    if not pending:
        return

    with jobs_path.open("a" if jobs_exist else "x", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        if not jobs_exist:
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
    allocation_count = (pending_count + capacity - 1) // capacity
    minimum_size, larger_count = divmod(pending_count, allocation_count)
    return [minimum_size + 1] * larger_count + [minimum_size] * (allocation_count - larger_count)


def _load_submitted_rows(jobs_path: Path) -> set[int]:
    with jobs_path.open(newline="", encoding="utf-8") as source:
        return {int(job["row"]) for job in csv.DictReader(source, delimiter="\t")}


app = typer.Typer(add_completion=False)
app.command()(candidates)
app.command()(workflows)


if __name__ == "__main__":
    app()

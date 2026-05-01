# pyright: strict

"""Durable benchmark plan creation and remote submission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.errors import SpiceOperatorError
from ..execution.session import ExecutionSession, open_execution_session
from .compilation import plan_benchmark
from .models import LoadedBenchmarkPlanEntry
from .runs import (
    BenchmarkSubmissionRecord,
    append_submission_jsonl,
    create_benchmark_run_dir,
    load_plan_jsonl,
    load_run_metadata,
    load_submission_jsonl,
    write_plan_jsonl,
)


@dataclass(frozen=True, slots=True)
class PlannedBenchmarkRun:
    run_dir: Path
    entry_count: int


@dataclass(frozen=True, slots=True)
class SubmittedBenchmarkWorkflow:
    record: BenchmarkSubmissionRecord
    run_dir: Path


def plan_benchmark_run(
    name: str,
    *,
    target: str,
    runs_root: Path,
) -> PlannedBenchmarkRun:
    entries = plan_benchmark(name)
    run_dir = create_benchmark_run_dir(name, target=target, runs_root=runs_root)
    write_plan_jsonl(run_dir, entries)
    return PlannedBenchmarkRun(run_dir=run_dir, entry_count=len(entries))


def submit_benchmark_run(run_dir: Path) -> list[SubmittedBenchmarkWorkflow]:
    metadata = load_run_metadata(run_dir)
    entries = load_plan_jsonl(run_dir)
    if load_submission_jsonl(run_dir):
        raise SpiceOperatorError(f"Benchmark run already has submissions: {run_dir}")
    session = open_execution_session(metadata.target)
    git_commit = session.remote_git_commit()
    return _submit_benchmark_entries(
        entries,
        run_dir=run_dir,
        session=session,
        git_commit=git_commit,
    )


def _submit_benchmark_entries(
    entries: list[LoadedBenchmarkPlanEntry],
    *,
    run_dir: Path,
    session: ExecutionSession,
    git_commit: str,
) -> list[SubmittedBenchmarkWorkflow]:
    submitted: dict[str, str] = {}
    records: list[SubmittedBenchmarkWorkflow] = []
    for entry in entries:
        dependency = compose_dependency(
            local_job_ids=[submitted[run_id] for run_id in entry.depends_on],
            external_dependencies=entry.external_dependencies,
        )
        submission = session.submit_workflow(
            entry.workflow,
            config=entry.config,
            dependency=dependency,
        )
        submitted[entry.run_id] = submission.job_id
        record = BenchmarkSubmissionRecord(
            run_id=entry.run_id,
            workflow=entry.workflow,
            job_id=submission.job_id,
            execution_ref=f"slurm:{submission.job_id}",
            git_commit=git_commit,
            dependency=dependency,
            log_path=str(submission.log_path),
        )
        append_submission_jsonl(run_dir, record)
        records.append(SubmittedBenchmarkWorkflow(record=record, run_dir=run_dir))
    return records


def compose_dependency(
    *,
    local_job_ids: list[str],
    external_dependencies: tuple[str, ...],
) -> str | None:
    parts = list(external_dependencies)
    if local_job_ids:
        parts.append("afterok:" + ":".join(local_job_ids))
    if not parts:
        return None
    return ",".join(parts)

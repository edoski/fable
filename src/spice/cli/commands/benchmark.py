"""Benchmark command routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..options import DEFAULT_REMOTE_TARGET, RemoteTargetOption

app = typer.Typer(
    help="Plan and submit benchmark matrices.",
    no_args_is_help=True,
)


@app.command(
    "plan",
    short_help="Print resolved benchmark plan JSONL.",
    help="Expand one benchmark YAML into validated resolved workflow config JSONL.",
)
def benchmark_plan_command(
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Benchmark spec name."),
    ],
) -> None:
    from ...config.benchmarks import plan_benchmark

    for entry in plan_benchmark(name):
        typer.echo(json.dumps(entry.to_json_dict(), sort_keys=True))


@app.command(
    "submit",
    short_help="Submit a benchmark plan remotely.",
    help="Submit one benchmark YAML through the configured remote execution target.",
)
def benchmark_submit_command(
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Benchmark spec name."),
    ],
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
) -> None:
    from ...benchmark_runs import (
        BenchmarkSubmissionRecord,
        append_submission_jsonl,
        create_benchmark_run_dir,
        resolve_remote_git_commit,
        write_plan_jsonl,
    )
    from ...config.benchmarks import plan_benchmark
    from ...execution.slurm_ssh import submit_execution_workflow

    entries = plan_benchmark(name)
    git_commit = resolve_remote_git_commit(target)
    run_dir = create_benchmark_run_dir(
        name,
        target=target,
        git_commit=git_commit,
    )
    write_plan_jsonl(run_dir, entries)
    submitted: dict[str, str] = {}
    for entry in entries:
        dependency = _compose_dependency(
            local_job_ids=[submitted[run_id] for run_id in entry.depends_on],
            external_dependencies=entry.external_dependencies,
        )
        submission = submit_execution_workflow(
            entry.workflow,
            config=entry.config,
            target_name=target,
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
        typer.echo(
            json.dumps(
                {**record.model_dump(mode="json"), "run_dir": str(run_dir)},
                sort_keys=True,
            )
        )


@app.command(
    "collect",
    short_help="Collect completed benchmark evaluations into the ledger.",
    help="Pull remote benchmark artifacts for a submitted run and optionally append ledger rows.",
)
def benchmark_collect_command(
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Benchmark spec name."),
    ],
    target: RemoteTargetOption = DEFAULT_REMOTE_TARGET,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Specific benchmark run directory. Defaults to latest run for NAME.",
        ),
    ] = None,
    ledger: Annotated[
        Path,
        typer.Option(
            "--ledger",
            help="Benchmark result ledger path.",
        ),
    ] = Path("benchmarks") / "results.csv",
    write: Annotated[
        bool,
        typer.Option(
            "--write",
            help="Append complete, non-duplicate evaluation rows to the ledger.",
        ),
    ] = False,
) -> None:
    from ...benchmark_runs import collect_benchmark_run, latest_benchmark_run_dir

    resolved_run_dir = run_dir or latest_benchmark_run_dir(name)
    records = collect_benchmark_run(
        run_dir=resolved_run_dir,
        target_name=target,
        ledger_path=ledger,
        write=write,
    )
    for record in records:
        typer.echo(json.dumps(record.model_dump(mode="json"), sort_keys=True))


def _compose_dependency(
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

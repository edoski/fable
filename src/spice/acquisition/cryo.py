"""Cryo command planning and execution."""

from __future__ import annotations

import math
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.config import AcquisitionConfig, ChainConfig, ProviderConfig
from ..core.console import NullReporter, Reporter
from .provider import redact_sensitive_text

CRYO_PROGRESS_POLL_INTERVAL_SECONDS = 0.5


@dataclass(slots=True)
class TimestampRange:
    start: int
    end: int

    def as_cryo_arg(self) -> str:
        return f"{self.start}:{self.end}"


@dataclass(slots=True)
class CryoRunResult:
    command: str
    completed_chunks: int
    expected_chunks: int | None


def history_range_for_required_blocks(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    *,
    required_history_blocks: int,
    window_start_timestamp: int,
) -> TimestampRange:
    if required_history_blocks <= 0:
        raise ValueError("required_history_blocks must be positive")
    block_count = required_history_blocks + acquisition.raw.chunk_size
    span_seconds = math.ceil(block_count * chain.block_time_seconds)
    return TimestampRange(
        start=window_start_timestamp - span_seconds,
        end=window_start_timestamp,
    )


def evaluation_range(start_timestamp: int, end_timestamp: int) -> TimestampRange:
    return TimestampRange(start=start_timestamp, end=end_timestamp)


def _existing_parquet_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for candidate in path.rglob("*.parquet") if candidate.is_file())


def _expected_chunk_count(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    timestamps: TimestampRange,
) -> int:
    span_seconds = max(1, timestamps.end - timestamps.start)
    approx_blocks = math.ceil(span_seconds / chain.block_time_seconds)
    return max(1, math.ceil(approx_blocks / acquisition.raw.chunk_size))


def _build_cryo_tokens(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    rpc_url: str,
    overwrite: bool = False,
) -> list[str]:
    tokens = [
        "cryo",
        "blocks",
        "--timestamps",
        timestamps.as_cryo_arg(),
        "--rpc",
        rpc_url,
        "--network-name",
        chain.name.value,
        "--include-columns",
        "all",
        "--output-dir",
        str(output_dir),
        "--requests-per-second",
        str(acquisition.raw.requests_per_second),
        "--max-concurrent-requests",
        str(acquisition.raw.max_concurrent_requests),
        "--max-concurrent-chunks",
        str(acquisition.raw.max_concurrent_chunks),
        "--chunk-size",
        str(acquisition.raw.chunk_size),
    ]
    if overwrite:
        tokens.append("--overwrite")
    return tokens


def build_cryo_args(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: ProviderConfig,
    overwrite: bool = False,
) -> list[str]:
    return _build_cryo_tokens(
        chain,
        acquisition,
        output_dir,
        timestamps,
        rpc_url=provider.endpoint_for(chain.name),
        overwrite=overwrite,
    )


def build_cryo_command(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: ProviderConfig,
    overwrite: bool = False,
) -> str:
    tokens = _build_cryo_tokens(
        chain,
        acquisition,
        output_dir,
        timestamps,
        rpc_url=provider.reference_for(chain.name),
        overwrite=overwrite,
    )
    return " ".join(shlex.quote(token) for token in tokens)


def _refresh_pull_progress(
    reporter: Reporter,
    *,
    task_id: int,
    output_dir: Path,
    baseline_chunk_count: int,
    completed_chunks: int,
    total_chunks: int | None,
    latest_output: str | None = None,
) -> int:
    current_completed_chunks = max(0, _existing_parquet_count(output_dir) - baseline_chunk_count)
    if latest_output is not None or current_completed_chunks != completed_chunks:
        reporter.update_task(task_id, completed=current_completed_chunks)
    if latest_output is not None:
        reporter.throttled_log(
            f"cryo-output:{output_dir}",
            latest_output,
            interval_seconds=10.0,
        )
    return current_completed_chunks


def _read_pull_output(
    process: subprocess.Popen[str],
    output_queue: queue.SimpleQueue[str | None],
) -> None:
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            output_queue.put(line)
    output_queue.put(None)


def _stream_pull_progress(
    process: subprocess.Popen[str],
    reporter: Reporter,
    *,
    task_id: int,
    output_dir: Path,
    baseline_chunk_count: int,
    total_chunks: int | None,
    provider: ProviderConfig,
) -> int:
    completed_chunks = 0
    output_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
    output_reader = threading.Thread(
        target=_read_pull_output,
        args=(process, output_queue),
        name=f"spice-cryo-output-{output_dir.name}",
    )
    output_reader.start()

    reached_eof = False
    while True:
        completed_chunks = _refresh_pull_progress(
            reporter,
            task_id=task_id,
            output_dir=output_dir,
            baseline_chunk_count=baseline_chunk_count,
            completed_chunks=completed_chunks,
            total_chunks=total_chunks,
        )
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                reached_eof = True
                continue
            completed_chunks = _refresh_pull_progress(
                reporter,
                task_id=task_id,
                output_dir=output_dir,
                baseline_chunk_count=baseline_chunk_count,
                completed_chunks=completed_chunks,
                total_chunks=total_chunks,
                latest_output=redact_sensitive_text(line.rstrip(), provider),
            )
        if reached_eof and process.poll() is not None:
            break
        time.sleep(CRYO_PROGRESS_POLL_INTERVAL_SECONDS)

    output_reader.join()
    return _refresh_pull_progress(
        reporter,
        task_id=task_id,
        output_dir=output_dir,
        baseline_chunk_count=baseline_chunk_count,
        completed_chunks=completed_chunks,
        total_chunks=total_chunks,
    )


def run_cryo(
    chain: ChainConfig,
    acquisition: AcquisitionConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: ProviderConfig,
    overwrite: bool = False,
    dry_run: bool = False,
    reporter: Reporter | None = None,
) -> CryoRunResult:
    reporter = reporter or NullReporter()
    expected_chunks = None if dry_run else _expected_chunk_count(chain, acquisition, timestamps)
    command = build_cryo_command(
        chain,
        acquisition,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
    )
    args = build_cryo_args(
        chain,
        acquisition,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
    )
    if dry_run:
        args.append("--dry")
        subprocess.run(args, check=True)
        reporter.log(f"dry run: {command}")
        return CryoRunResult(command=command, completed_chunks=0, expected_chunks=expected_chunks)

    baseline_chunk_count = _existing_parquet_count(output_dir)
    task_id = reporter.start_task(
        f"pull {chain.name.value}:{output_dir.name}",
        total=expected_chunks,
        unit="chunks",
    )
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    latest_completed_chunks = _stream_pull_progress(
        process,
        reporter,
        task_id=task_id,
        output_dir=output_dir,
        baseline_chunk_count=baseline_chunk_count,
        total_chunks=expected_chunks,
        provider=provider,
    )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)
    reporter.finish_task(task_id, message=str(output_dir))
    return CryoRunResult(
        command=command,
        completed_chunks=latest_completed_chunks,
        expected_chunks=expected_chunks,
    )

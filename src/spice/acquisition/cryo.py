"""Cryo command planning and execution."""

from __future__ import annotations

import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..core.config import ChainConfig, ExperimentConfig, PullConfig
from ..core.console import NullReporter, Reporter
from ..core.constants import EVALUATION_END_TS, EVALUATION_START_TS
from .rpc_providers import RpcProvider, redact_sensitive_text


@dataclass(slots=True)
class TimestampRange:
    start: int
    end: int

    def as_cryo_arg(self) -> str:
        return f"{self.start}:{self.end}"


@dataclass(slots=True)
class CryoCommandPlan:
    chain: str
    history_range: TimestampRange
    evaluation_range: TimestampRange
    history_output_dir: Path
    evaluation_output_dir: Path
    command: str


@dataclass(slots=True)
class CryoRunResult:
    command: str
    completed_chunks: int
    expected_chunks: int | None


def history_range_for_chain(chain: ChainConfig) -> TimestampRange:
    span = chain.history_days * 24 * 60 * 60
    return TimestampRange(start=EVALUATION_START_TS - span, end=EVALUATION_START_TS)


def evaluation_range() -> TimestampRange:
    return TimestampRange(start=EVALUATION_START_TS, end=EVALUATION_END_TS)


def _existing_parquet_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for candidate in path.rglob("*.parquet") if candidate.is_file())


def _expected_chunk_count(chain: ChainConfig, pull: PullConfig, timestamps: TimestampRange) -> int:
    span_seconds = max(1, timestamps.end - timestamps.start)
    approx_blocks = math.ceil(span_seconds / chain.block_time_seconds)
    return max(1, math.ceil(approx_blocks / pull.chunk_size))


def _build_cryo_tokens(
    chain: ChainConfig,
    pull: PullConfig,
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
        str(pull.requests_per_second),
        "--max-concurrent-requests",
        str(pull.max_concurrent_requests),
        "--max-concurrent-chunks",
        str(pull.max_concurrent_chunks),
        "--chunk-size",
        str(pull.chunk_size),
    ]
    if overwrite:
        tokens.append("--overwrite")
    return tokens


def build_cryo_args(
    chain: ChainConfig,
    pull: PullConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: RpcProvider,
    overwrite: bool = False,
) -> list[str]:
    return _build_cryo_tokens(
        chain,
        pull,
        output_dir,
        timestamps,
        rpc_url=provider.url_for(chain.name),
        overwrite=overwrite,
    )


def build_cryo_command(
    chain: ChainConfig,
    pull: PullConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: RpcProvider,
    overwrite: bool = False,
) -> str:
    tokens = _build_cryo_tokens(
        chain,
        pull,
        output_dir,
        timestamps,
        rpc_url=provider.reference_for(chain.name),
        overwrite=overwrite,
    )
    return " ".join(shlex.quote(token) for token in tokens)


def run_cryo(
    chain: ChainConfig,
    pull: PullConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    provider: RpcProvider,
    overwrite: bool = False,
    dry_run: bool = False,
    reporter: Reporter | None = None,
) -> CryoRunResult:
    reporter = reporter or NullReporter()
    expected_chunks = None if dry_run else _expected_chunk_count(chain, pull, timestamps)
    command = build_cryo_command(
        chain,
        pull,
        output_dir,
        timestamps,
        provider=provider,
        overwrite=overwrite,
    )
    args = build_cryo_args(
        chain,
        pull,
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
    reporter.start_pull(
        label=f"pull {chain.name.value}:{output_dir.name} (approx chunks)",
        total_chunks=expected_chunks,
    )
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    latest_completed_chunks = 0
    assert process.stdout is not None
    for line in process.stdout:
        latest_completed_chunks = max(0, _existing_parquet_count(output_dir) - baseline_chunk_count)
        reporter.update_pull(
            completed_chunks=latest_completed_chunks,
            total_chunks=expected_chunks,
            latest_output=redact_sensitive_text(line.rstrip(), provider),
        )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)
    latest_completed_chunks = max(0, _existing_parquet_count(output_dir) - baseline_chunk_count)
    reporter.update_pull(completed_chunks=latest_completed_chunks, total_chunks=expected_chunks)
    reporter.finish_pull(output_dir=output_dir)
    return CryoRunResult(
        command=command,
        completed_chunks=latest_completed_chunks,
        expected_chunks=expected_chunks,
    )


def build_pull_plan(config: ExperimentConfig, *, provider: RpcProvider) -> list[CryoCommandPlan]:
    plans: list[CryoCommandPlan] = []
    for chain in config.chains:
        history_output_dir = config.output_root / "raw" / chain.name.value / "history"
        evaluation_output_dir = config.output_root / "raw" / chain.name.value / "evaluation"
        history = history_range_for_chain(chain)
        evaluation = evaluation_range()
        command = "\n".join(
            [
                build_cryo_command(
                    chain,
                    config.pull,
                    history_output_dir,
                    history,
                    provider=provider,
                ),
                build_cryo_command(
                    chain,
                    config.pull,
                    evaluation_output_dir,
                    evaluation,
                    provider=provider,
                ),
            ]
        )
        plans.append(
            CryoCommandPlan(
                chain=chain.name.value,
                history_range=history,
                evaluation_range=evaluation,
                history_output_dir=history_output_dir,
                evaluation_output_dir=evaluation_output_dir,
                command=command,
            )
        )
    return plans

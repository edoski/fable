"""Helpers for planning and validating cryo pulls."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spice_temporal.config import ChainConfig, ExperimentConfig, PullConfig
from spice_temporal.constants import EVALUATION_END_TS, EVALUATION_START_TS
from spice_temporal.env import ALCHEMY_RPC_TEMPLATE, resolve_rpc_url


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


def history_range_for_chain(chain: ChainConfig) -> TimestampRange:
    span = chain.history_days * 24 * 60 * 60
    return TimestampRange(start=EVALUATION_START_TS - span, end=EVALUATION_START_TS)


def evaluation_range() -> TimestampRange:
    return TimestampRange(start=EVALUATION_START_TS, end=EVALUATION_END_TS)


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
        chain.name,
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
    overwrite: bool = False,
) -> list[str]:
    return _build_cryo_tokens(
        chain,
        pull,
        output_dir,
        timestamps,
        rpc_url=resolve_rpc_url(chain.name),
        overwrite=overwrite,
    )


def build_cryo_command(
    chain: ChainConfig,
    pull: PullConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    overwrite: bool = False,
) -> str:
    rpc_template = ALCHEMY_RPC_TEMPLATE[chain.name].replace("{api_key}", "$ALCHEMY_API_KEY")
    tokens = _build_cryo_tokens(
        chain,
        pull,
        output_dir,
        timestamps,
        rpc_url=rpc_template,
        overwrite=overwrite,
    )
    return " ".join(shlex.quote(token) for token in tokens)


def run_cryo(
    chain: ChainConfig,
    pull: PullConfig,
    output_dir: Path,
    timestamps: TimestampRange,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    args = build_cryo_args(
        chain,
        pull,
        output_dir,
        timestamps,
        overwrite=overwrite,
    )
    if dry_run:
        args.append("--dry")
    return subprocess.run(args, check=True, text=True, capture_output=True)


def build_pull_plan(config: ExperimentConfig) -> list[CryoCommandPlan]:
    plans: list[CryoCommandPlan] = []
    for chain in config.chains:
        history_output_dir = config.output_root / "raw" / chain.name / "history"
        evaluation_output_dir = config.output_root / "raw" / chain.name / "evaluation"
        history = history_range_for_chain(chain)
        evaluation = evaluation_range()
        command = "\n".join(
            [
                build_cryo_command(chain, config.pull, history_output_dir, history),
                build_cryo_command(chain, config.pull, evaluation_output_dir, evaluation),
            ]
        )
        plans.append(
            CryoCommandPlan(
                chain=chain.name,
                history_range=history,
                evaluation_range=evaluation,
                history_output_dir=history_output_dir,
                evaluation_output_dir=evaluation_output_dir,
                command=command,
            )
        )
    return plans

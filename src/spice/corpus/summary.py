"""Corpus-owned workflow summary builders."""

from __future__ import annotations

from datetime import UTC, datetime

from ..acquisition.rpc import BlockPullPlan
from ..config import AcquireConfig
from ..corpus.builders import DatasetBuildOutcome
from ..features import FeaturePrerequisites
from ..storage.layout import resolve_workflow_paths
from ..temporal.contracts import CompiledProblemContract


def acquire_dry_run_sections(
    config: AcquireConfig,
    *,
    contract: CompiledProblemContract,
    history_window_seconds: int,
    history_plan: BlockPullPlan,
    evaluation_plan: BlockPullPlan,
) -> list[tuple[str, list[tuple[str, str]]]]:
    paths = resolve_workflow_paths(config)
    return [
        (
            "dataset",
            [
                ("name", config.dataset.name),
                ("storage id", paths.corpus_id),
                ("chain", config.chain.name),
                ("problem", config.problem.id),
                ("compiler", contract.compiler_id),
                ("feature set", config.feature_set.id),
                ("feature family", contract.feature_family_id),
                ("evaluation date", str(config.dataset.evaluation_date)),
                (
                    "feature prerequisites",
                    _feature_prerequisites_string(contract.feature_prerequisites),
                ),
                ("lookback", f"{contract.lookback_seconds}s"),
                ("history window", f"{history_window_seconds}s"),
            ],
        ),
        (
            "history",
            _planned_window_rows(
                start_timestamp=history_plan.window.start,
                end_timestamp=history_plan.window.end,
                expected_rows=history_plan.expected_rows,
                expected_files=history_plan.expected_files,
            ),
        ),
        (
            "evaluation",
            _planned_window_rows(
                start_timestamp=evaluation_plan.window.start,
                end_timestamp=evaluation_plan.window.end,
                expected_rows=evaluation_plan.expected_rows,
                expected_files=evaluation_plan.expected_files,
            ),
        ),
    ]


def acquisition_summary_sections(
    config: AcquireConfig,
    *,
    provider_name: str,
    history_outcome: DatasetBuildOutcome,
    history_row_count: int,
    history_file_count: int,
    evaluation_outcome: DatasetBuildOutcome,
    evaluation_row_count: int,
    evaluation_file_count: int,
) -> list[tuple[str, list[tuple[str, str]]]]:
    paths = resolve_workflow_paths(config)
    return [
        (
            "dataset",
            [
                ("name", config.dataset.name),
                ("storage id", paths.corpus_id),
                ("chain", config.chain.name),
                ("problem", config.problem.id),
                ("feature set", config.feature_set.id),
                ("feature family", config.feature_set.family.id),
                ("provider", provider_name),
            ],
        ),
        (
            "history",
            _final_window_rows(
                outcome=history_outcome,
                row_count=history_row_count,
                file_count=history_file_count,
            ),
        ),
        (
            "evaluation",
            _final_window_rows(
                outcome=evaluation_outcome,
                row_count=evaluation_row_count,
                file_count=evaluation_file_count,
            ),
        ),
    ]


def _format_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _format_duration(start_timestamp: int, end_timestamp: int) -> str:
    remaining = max(0, end_timestamp - start_timestamp)
    units = (
        ("d", 24 * 60 * 60),
        ("h", 60 * 60),
        ("m", 60),
        ("s", 1),
    )
    parts: list[str] = []
    for suffix, size in units:
        if remaining < size and parts:
            continue
        value, remaining = divmod(remaining, size)
        if value > 0 or not parts:
            parts.append(f"{value}{suffix}")
        if len(parts) == 2:
            break
    return " ".join(parts)


def _format_count(value: int, singular: str, plural: str | None = None) -> str:
    unit = singular if value == 1 else (plural or f"{singular}s")
    return f"{value:,} {unit}"


def _feature_prerequisites_string(prerequisites: FeaturePrerequisites) -> str:
    return " ".join(
        [
            f"history={prerequisites.history_seconds}s",
            f"warmup={prerequisites.warmup_rows} rows",
        ]
    )


def _planned_window_rows(
    *,
    start_timestamp: int,
    end_timestamp: int,
    expected_rows: int,
    expected_files: int,
) -> list[tuple[str, str]]:
    return [
        ("window", f"{_format_timestamp(start_timestamp)} -> {_format_timestamp(end_timestamp)}"),
        ("duration", _format_duration(start_timestamp, end_timestamp)),
        (
            "planned",
            f"{_format_count(expected_rows, 'block')} in {_format_count(expected_files, 'file')}",
        ),
    ]


def _final_window_rows(
    *,
    outcome: DatasetBuildOutcome,
    row_count: int,
    file_count: int,
) -> list[tuple[str, str]]:
    return [
        ("status", outcome.value),
        ("blocks", _format_count(row_count, "block")),
        ("files", _format_count(file_count, "file")),
    ]

"""Canonical block dataset acquisition workflow."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ..acquisition.datasets import (
    DatasetBuildOutcome,
    build_history_plan,
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..acquisition.metadata import (
    build_dataset_metadata,
    load_dataset_metadata,
    merge_providers,
    provider_metadata,
)
from ..acquisition.rpc import RpcController, Web3BlockClient, evaluation_range
from ..config import AcquireConfig
from ..core.console import Reporter
from ..core.files import promote_paths_atomic
from ..core.json import write_json
from ..planning.geometry import derive_dataset_geometry
from ._shared import managed_workflow


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


def _format_build_outcome(outcome: DatasetBuildOutcome) -> str:
    return outcome.value


def _final_window_rows(
    *,
    outcome: DatasetBuildOutcome,
    row_count: int,
    file_count: int,
) -> list[tuple[str, str]]:
    return [
        ("status", _format_build_outcome(outcome)),
        ("blocks", _format_count(row_count, "block")),
        ("files", _format_count(file_count, "file")),
    ]


def _workflow_facts(config: AcquireConfig) -> list[tuple[str, str]]:
    return [
        ("dataset", config.dataset.id),
        ("chain", config.chain.name),
        ("provider", config.provider.name),
    ]


def run(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    asyncio.run(_run_async(config, reporter=reporter))


async def _run_async(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    history_dir = config.paths.history_dir
    evaluation_dir = config.paths.evaluation_dir
    metadata_path = config.paths.dataset_metadata_path
    rpc_controller = RpcController.from_config(config.acquisition)

    geometry = derive_dataset_geometry(
        lookback_seconds=config.dataset.temporal.lookback_seconds,
        max_delay_seconds=config.dataset.temporal.max_delay_seconds,
        block_time_seconds=config.chain.runtime.block_time_seconds,
        history_context_blocks=config.dataset.history_context_blocks,
    )
    required_history_blocks = geometry.required_block_count(config.effective_history_sample_budget)
    evaluation_window = evaluation_range(
        config.evaluation_window_start_timestamp,
        config.evaluation_window_end_timestamp,
    )
    chain_label = config.chain.name

    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name}-{config.provider.name}",
        reporter=reporter,
    ) as session:
        session.runtime.configure_workflow("acquire", _workflow_facts(config))
        history_reporter = session.runtime.stage_reporter(
            "history",
            label="history",
            status="pending",
            running_status="pulling",
        )
        evaluation_reporter = session.runtime.stage_reporter(
            "evaluation",
            label="evaluation",
            status="pending",
            running_status="pulling",
        )
        metadata_reporter = session.runtime.stage_reporter(
            "metadata",
            label="metadata",
            status="pending",
            running_status="writing",
        )
        block_client = Web3BlockClient(config.provider, config.chain)
        try:
            evaluation_plan = await block_client.plan_window(
                evaluation_window,
                chunk_size=config.acquisition.chunk_size,
            )
            history_plan = await build_history_plan(
                config=config,
                block_client=block_client,
                required_history_blocks=required_history_blocks,
            )
            session.runtime.set_stage_state(
                "history",
                total=history_plan.expected_rows,
                completed=0,
                unit="blocks",
                message="waiting",
            )
            session.runtime.set_stage_state(
                "evaluation",
                total=evaluation_plan.expected_rows,
                completed=0,
                unit="blocks",
                message="waiting for history",
            )

            if config.acquisition.dry_run:
                session.runtime.log_sectioned_summary(
                    "acquire dry run",
                    [
                        (
                            "dataset",
                            [
                                ("id", config.dataset.id),
                                ("chain", chain_label),
                                ("evaluation date", str(config.dataset.evaluation_date)),
                                (
                                    "required history",
                                    _format_count(required_history_blocks, "block"),
                                ),
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
                                start_timestamp=evaluation_window.start,
                                end_timestamp=evaluation_window.end,
                                expected_rows=evaluation_plan.expected_rows,
                                expected_files=evaluation_plan.expected_files,
                            ),
                        ),
                    ],
                )
                return

            existing_metadata = load_dataset_metadata(metadata_path)
            try:
                config.paths.dataset_root.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(
                    dir=config.paths.dataset_root.parent,
                    prefix=f".{config.dataset.id}.acquire.",
                ) as temp_root_name:
                    temp_root = Path(temp_root_name)
                    history_result = await ensure_history_dataset(
                        config=config,
                        block_client=block_client,
                        output_dir=history_dir,
                        working_dir=temp_root,
                        history_plan=history_plan,
                        required_history_blocks=required_history_blocks,
                        rpc_controller=rpc_controller,
                        reporter=history_reporter,
                    )
                    session.runtime.set_stage_state(
                        "history",
                        status=history_result.outcome.value,
                        total=history_result.validation.row_count,
                        completed=history_result.validation.row_count,
                        unit="blocks",
                        message=f"{history_result.file_count:,} files",
                    )
                    evaluation_result = await ensure_evaluation_dataset(
                        config=config,
                        block_client=block_client,
                        output_dir=evaluation_dir,
                        working_dir=temp_root,
                        evaluation_plan=evaluation_plan,
                        rpc_controller=rpc_controller,
                        reporter=evaluation_reporter,
                    )
                    session.runtime.set_stage_state(
                        "evaluation",
                        status=evaluation_result.outcome.value,
                        total=evaluation_result.validation.row_count,
                        completed=evaluation_result.validation.row_count,
                        unit="blocks",
                        message=f"{evaluation_result.file_count:,} files",
                    )
                    providers = list(existing_metadata.providers) if existing_metadata else []
                    current_provider = provider_metadata(config)
                    if (
                        not providers
                        or history_result.pulled_blocks
                        or evaluation_result.pulled_blocks
                    ):
                        providers = merge_providers(providers, current_provider)
                    metadata = build_dataset_metadata(
                        config=config,
                        history_dir=history_dir,
                        evaluation_dir=evaluation_dir,
                        history_request_start_timestamp=history_plan.window.start,
                        history_request_end_timestamp=history_plan.window.end,
                        evaluation_request_start_timestamp=evaluation_window.start,
                        evaluation_request_end_timestamp=evaluation_window.end,
                        providers=providers,
                        history_validation=history_result.validation,
                        evaluation_validation=evaluation_result.validation,
                        acquisition_runtime=rpc_controller.snapshot(),
                    )
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_task = metadata_reporter.start_task("write dataset metadata")
                    metadata_tmp_path = temp_root / ".spice" / "metadata.json"
                    write_json(metadata_tmp_path, metadata)
                    promotions: list[tuple[Path, Path]] = []
                    if history_result.promote_dir is not None:
                        promotions.append((history_dir, history_result.promote_dir))
                    if evaluation_result.promote_dir is not None:
                        promotions.append((evaluation_dir, evaluation_result.promote_dir))
                    promotions.append((metadata_path, metadata_tmp_path))
                    promote_paths_atomic(promotions)
                    metadata_reporter.finish_task(
                        metadata_task,
                        message=str(metadata_path),
                        silent=True,
                    )
            except KeyboardInterrupt:
                session.reporter.log(
                    "acquire interrupted; temporary outputs removed, canonical dataset preserved",
                    level="warning",
                )
                raise
            except Exception:
                session.reporter.log(
                    "acquire failed; temporary outputs removed, canonical dataset preserved",
                    level="warning",
                )
                raise
            session.runtime.log_sectioned_summary(
                "acquisition summary",
                [
                    (
                        "dataset",
                        [
                            ("id", config.dataset.id),
                            ("chain", chain_label),
                        ],
                    ),
                    (
                        "history",
                        _final_window_rows(
                            outcome=history_result.outcome,
                            row_count=history_result.validation.row_count,
                            file_count=history_result.file_count,
                        ),
                    ),
                    (
                        "evaluation",
                        _final_window_rows(
                            outcome=evaluation_result.outcome,
                            row_count=evaluation_result.validation.row_count,
                            file_count=evaluation_result.file_count,
                        ),
                    ),
                ],
            )
        finally:
            await block_client.close()

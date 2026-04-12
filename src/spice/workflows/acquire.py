"""Canonical block dataset acquisition workflow."""

from __future__ import annotations

import asyncio
import signal
import threading
import weakref
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..acquisition.datasets import (
    DatasetBuildOutcome,
    build_history_plan,
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..acquisition.metadata import (
    build_acquire_run_record,
    build_dataset_summary,
    provider_metadata,
)
from ..acquisition.rpc import RpcController, Web3BlockClient, evaluation_range
from ..config import AcquireConfig
from ..core.console import Reporter
from ..core.files import promote_paths_atomic, prune_empty_directories
from ..planning.contracts import resolve_task_contract
from ..state.dataset import write_dataset_state
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
        ("task", config.task.id),
        ("feature set", config.feature_set.id),
        ("provider", config.provider.name),
    ]


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Default executor variant that doesn't block interpreter shutdown on SIGINT."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return
        thread_name = f"{self._thread_name_prefix or self}_{num_threads}"
        thread = threading.Thread(
            name=thread_name,
            target=_worker,
            args=(
                weakref.ref(self, weakref_cb),
                self._work_queue,
                self._initializer,
                self._initargs,
            ),
        )
        thread.daemon = True
        thread.start()
        self._threads.add(thread)
        # Skip concurrent.futures' global exit registry so a cancelled acquire
        # doesn't hang in interpreter shutdown waiting for RPC worker threads.


def _run_async_interruptibly(coro: Coroutine[Any, Any, None]) -> None:
    loop = asyncio.new_event_loop()
    executor = _DaemonThreadPoolExecutor(thread_name_prefix="spice-asyncio")
    loop.set_default_executor(executor)
    task = loop.create_task(coro)
    interrupted = False
    previous_sigint = None

    def _handle_sigint(signum, frame) -> None:
        del signum, frame
        nonlocal interrupted
        interrupted = True
        if not task.done():
            loop.call_soon_threadsafe(task.cancel)

    try:
        with suppress(ValueError):
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _handle_sigint)
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            interrupted = True
        except KeyboardInterrupt:
            interrupted = True
            if not task.done():
                task.cancel()
                with suppress(BaseException):
                    loop.run_until_complete(task)
    finally:
        if previous_sigint is not None:
            with suppress(ValueError):
                signal.signal(signal.SIGINT, previous_sigint)
        pending = [
            pending_task
            for pending_task in asyncio.all_tasks(loop)
            if not pending_task.done()
        ]
        for pending_task in pending:
            pending_task.cancel()
        if pending and not interrupted:
            with suppress(BaseException):
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        with suppress(BaseException):
            loop.run_until_complete(loop.shutdown_asyncgens())
        executor.shutdown(wait=False, cancel_futures=True)
        asyncio.set_event_loop(None)
        loop.close()


def run(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    try:
        _run_async_interruptibly(_run_async(config, reporter=reporter))
    except KeyboardInterrupt:
        return None


async def _run_async(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    history_dir = config.paths.history_dir
    evaluation_dir = config.paths.evaluation_dir
    state_db_path = config.paths.dataset_state_db
    rpc_controller = RpcController.from_config(config.acquisition)
    contract = resolve_task_contract(
        chain=config.chain,
        task=config.task,
        feature_set=config.feature_set,
    )
    required_history_blocks = contract.required_history_blocks
    evaluation_window = evaluation_range(
        config.evaluation_window_start_timestamp,
        config.evaluation_window_end_timestamp,
    )
    chain_label = config.chain.name

    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name}-{config.task.id}-{config.provider.name}",
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
            )
            session.runtime.set_stage_state(
                "evaluation",
                total=evaluation_plan.expected_rows,
                completed=0,
                unit="blocks",
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
                                ("task", config.task.id),
                                ("feature set", config.feature_set.id),
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

            current_provider = provider_metadata(config)
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
                    )
                    summary = build_dataset_summary(
                        config=config,
                        history_request_start_timestamp=history_plan.window.start,
                        history_request_end_timestamp=history_plan.window.end,
                        evaluation_request_start_timestamp=evaluation_window.start,
                        evaluation_request_end_timestamp=evaluation_window.end,
                        provider=current_provider,
                        history_validation=history_result.validation,
                        evaluation_validation=evaluation_result.validation,
                    )
                    acquire_run = build_acquire_run_record(
                        config=config,
                        provider=current_provider,
                        contract=contract,
                        acquisition_runtime=rpc_controller.snapshot(),
                    )
                    temp_state_db = temp_root / ".spice" / "state.sqlite"
                    write_dataset_state(
                        temp_state_db,
                        summary=summary,
                        acquire_run=acquire_run,
                    )
                    promotions: list[tuple[Path, Path]] = []
                    if history_result.promote_dir is not None:
                        promotions.append((history_dir, history_result.promote_dir))
                    if evaluation_result.promote_dir is not None:
                        promotions.append((evaluation_dir, evaluation_result.promote_dir))
                    promotions.append((state_db_path, temp_state_db))
                    promote_paths_atomic(promotions)
            except (KeyboardInterrupt, asyncio.CancelledError):
                session.reporter.close()
                prune_empty_directories(
                    config.paths.dataset_root,
                    stop_at=config.paths.dataset_root.parent.parent,
                )
                session.reporter.log(
                    "acquire cancelled; partial download removed",
                    level="warning",
                )
                raise
            except Exception:
                session.reporter.close()
                session.reporter.log(
                    "acquire failed; partial download removed",
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
                            ("task", config.task.id),
                            ("feature set", config.feature_set.id),
                            ("state", str(state_db_path)),
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

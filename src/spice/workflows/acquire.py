"""Canonical block corpus acquisition workflow."""

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
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..acquisition.datasets import (
    DatasetBuildOutcome,
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..acquisition.metadata import (
    build_acquire_run_record,
    build_dataset_summary,
    provider_metadata,
)
from ..acquisition.rpc import RpcController, TimestampRange, Web3BlockClient, evaluation_range
from ..config import AcquireConfig
from ..core.console import Reporter
from ..core.files import promote_paths_atomic, prune_empty_directories
from ..data.datasets import build_temporal_store
from ..data.io import load_block_frame
from ..features import FeatureSelection, build_feature_table, make_feature_selection
from ..planning.contracts import resolve_task_contract
from ..planning.geometry import DelayWindow
from ..state.catalog import upsert_dataset_record
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
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("task", config.task.id),
        ("feature set", config.feature_set.id),
        ("provider", config.provider.name),
    ]


def _count_valid_history_samples(
    *,
    history_dir: Path,
    selection: FeatureSelection,
    window: DelayWindow,
) -> int:
    blocks = load_block_frame(history_dir).sort("block_number")
    feature_table = build_feature_table(blocks, selection=selection)
    store = build_temporal_store(feature_table, window=window)
    return store.n_samples


def _initial_history_window_seconds(
    *,
    required_history_seconds: int,
    sample_count: int,
    max_supported_delay_seconds: int,
    estimated_block_interval_seconds: float,
) -> int:
    return max(
        required_history_seconds + max_supported_delay_seconds,
        required_history_seconds
        + max_supported_delay_seconds
        + ceil(sample_count * estimated_block_interval_seconds),
    )


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
    state_db_path = config.paths.corpus_state_db
    rpc_controller = RpcController.from_config(config.acquisition)
    contract = resolve_task_contract(
        task=config.task,
        feature_set=config.feature_set,
    )
    selection = make_feature_selection(
        config.feature_set.id,
        tuple(config.feature_set.outputs),
    )
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

        def _update_stage(key: str, status: str, message: str | None = None) -> None:
            session.runtime.set_stage_state(key, status=status, message=message)

        def _update_history_stage(status: str, message: str | None = None) -> None:
            _update_stage("history", status, message)

        def _update_evaluation_stage(status: str, message: str | None = None) -> None:
            _update_stage("evaluation", status, message)

        block_client = Web3BlockClient(config.provider, config.chain)
        try:
            _update_history_stage("planning", "resolving window")
            _update_evaluation_stage("planning", "resolving window")
            evaluation_plan = await block_client.plan_window(
                evaluation_window,
                chunk_size=config.acquisition.chunk_size,
            )
            estimated_block_interval_seconds = await block_client.estimate_recent_block_interval()
            history_window_seconds = _initial_history_window_seconds(
                required_history_seconds=contract.required_history_seconds,
                sample_count=contract.sample_count,
                max_supported_delay_seconds=contract.max_supported_delay_seconds,
                estimated_block_interval_seconds=estimated_block_interval_seconds,
            )
            history_start_timestamp = max(
                0,
                config.history_window_end_timestamp - history_window_seconds,
            )
            history_plan = await block_client.plan_window(
                TimestampRange(
                    start=history_start_timestamp,
                    end=config.history_window_end_timestamp,
                ),
                chunk_size=config.acquisition.chunk_size,
            )
            session.runtime.set_stage_state(
                "history",
                status="planning",
                total=history_plan.expected_rows,
                completed=0,
                unit="blocks",
                message="checking existing dataset",
            )
            session.runtime.set_stage_state(
                "evaluation",
                status="pending",
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
                                ("name", config.dataset.name),
                                ("storage id", config.paths.corpus_id),
                                ("chain", chain_label),
                                ("task", config.task.id),
                                ("feature set", config.feature_set.id),
                                ("evaluation date", str(config.dataset.evaluation_date)),
                                ("feature history", f"{contract.feature_history_seconds}s"),
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
            config.paths.corpus_root.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                dir=config.paths.corpus_root.parent,
                prefix=f".{config.paths.corpus_id}.acquire.",
            ) as temp_root_name:
                temp_root = Path(temp_root_name)
                while True:
                    history_result = await ensure_history_dataset(
                        config=config,
                        block_client=block_client,
                        output_dir=history_dir,
                        working_dir=temp_root,
                        history_plan=history_plan,
                        rpc_controller=rpc_controller,
                        reporter=history_reporter,
                        stage_update=_update_history_stage,
                    )
                    valid_anchor_samples = _count_valid_history_samples(
                        history_dir=history_result.path,
                        selection=selection,
                        window=contract.capability_window,
                    )
                    if valid_anchor_samples >= contract.sample_count:
                        break
                    history_window_seconds *= 2
                    history_start_timestamp = max(
                        0,
                        config.history_window_end_timestamp - history_window_seconds,
                    )
                    history_plan = await block_client.plan_window(
                        TimestampRange(
                            start=history_start_timestamp,
                            end=config.history_window_end_timestamp,
                        ),
                        chunk_size=config.acquisition.chunk_size,
                    )
                session.runtime.set_stage_state(
                    "history",
                    status=history_result.outcome.value,
                    total=history_result.validation.row_count,
                    completed=history_result.validation.row_count,
                    unit="blocks",
                    message=(
                        f"{history_result.file_count:,} files, "
                        f"anchors={valid_anchor_samples:,}"
                    ),
                )
                evaluation_result = await ensure_evaluation_dataset(
                    config=config,
                    block_client=block_client,
                    output_dir=evaluation_dir,
                    working_dir=temp_root,
                    evaluation_plan=evaluation_plan,
                    rpc_controller=rpc_controller,
                    reporter=evaluation_reporter,
                    stage_update=_update_evaluation_stage,
                )
                session.runtime.set_stage_state(
                    "evaluation",
                    status=evaluation_result.outcome.value,
                    total=evaluation_result.validation.row_count,
                    completed=evaluation_result.validation.row_count,
                    unit="blocks",
                    message=f"{evaluation_result.file_count:,} files",
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
                    acquired_history_window_seconds=history_window_seconds,
                    valid_anchor_samples=valid_anchor_samples,
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
                upsert_dataset_record(
                    config.paths.catalog_db,
                    dataset_id=config.paths.corpus_id,
                    dataset_name=config.dataset.name,
                    chain_name=config.chain.name,
                    provider_name=current_provider.name,
                    root_path=config.paths.corpus_root,
                    state_db_path=state_db_path,
                )
            session.runtime.log_sectioned_summary(
                "acquisition summary",
                [
                    (
                        "dataset",
                        [
                            ("name", config.dataset.name),
                            ("storage id", config.paths.corpus_id),
                            ("chain", chain_label),
                            ("task", config.task.id),
                            ("feature set", config.feature_set.id),
                            ("provider", current_provider.name),
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
        except (KeyboardInterrupt, asyncio.CancelledError):
            session.reporter.close()
            prune_empty_directories(
                config.paths.corpus_root,
                stop_at=config.paths.corpus_root.parent.parent,
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
        finally:
            await block_client.close()

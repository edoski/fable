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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from ..acquisition.rpc import RpcController, TimestampRange, Web3BlockClient, evaluation_range
from ..config import AcquireConfig
from ..core.files import promote_paths_atomic, prune_empty_directories
from ..core.reporting import Reporter, StageMetricDescriptor
from ..corpus.builders import (
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from ..corpus.io import load_block_frame
from ..corpus.metadata import (
    build_acquire_run_record,
    build_dataset_summary,
    provider_metadata,
)
from ..corpus.summary import acquire_dry_run_sections, acquisition_summary_sections
from ..features import CompiledFeatureContract, compile_feature_contract
from ..storage.catalog import upsert_dataset_record
from ..storage.corpus import write_dataset_state
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract
from ._shared import managed_workflow

_RPC_STAGE_METRICS: tuple[StageMetricDescriptor, ...] = (
    StageMetricDescriptor(id="batch", label="batch", width=7),
    StageMetricDescriptor(id="conc", label="conc", width=5),
)


def _workflow_facts(config: AcquireConfig) -> list[tuple[str, str]]:
    return [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("feature set", config.feature_set.id),
        ("provider", config.provider.name),
    ]


def _count_valid_history_samples(
    *,
    history_dir: Path,
    feature_contract: CompiledFeatureContract,
    contract: CompiledProblemContract,
) -> int:
    blocks = load_block_frame(history_dir).sort("block_number")
    feature_table = feature_contract.build_table(blocks)
    return contract.count_valid_capability_samples(feature_table)


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Default executor variant that doesn't block interpreter shutdown on SIGINT."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            cast(Any, q).put(None)

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
        cast(set[threading.Thread], self._threads).add(thread)
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
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
    )
    evaluation_window = evaluation_range(
        config.evaluation_window_start_timestamp,
        config.evaluation_window_end_timestamp,
    )

    with managed_workflow(
        config,
        run_name=f"acquire-{config.chain.name}-{config.problem.id}-{config.provider.name}",
        reporter=reporter,
    ) as session:
        session.runtime.configure_workflow("acquire", _workflow_facts(config))
        history_reporter = session.runtime.stage_reporter(
            "history",
            label="history",
            status="pending",
            running_status="pulling",
            metric_descriptors=_RPC_STAGE_METRICS,
        )
        evaluation_reporter = session.runtime.stage_reporter(
            "evaluation",
            label="evaluation",
            status="pending",
            running_status="pulling",
            metric_descriptors=_RPC_STAGE_METRICS,
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
            history_window_seconds = contract.initial_history_window_seconds(
                estimated_block_interval_seconds,
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
                    acquire_dry_run_sections(
                        config,
                        contract=contract,
                        history_window_seconds=history_window_seconds,
                        history_plan=history_plan,
                        evaluation_plan=evaluation_plan,
                    ),
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
                        feature_contract=feature_contract,
                        contract=contract,
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
                    feature_contract=feature_contract,
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
                acquisition_summary_sections(
                    config,
                    provider_name=current_provider.name,
                    history_outcome=history_result.outcome,
                    history_row_count=history_result.validation.row_count,
                    history_file_count=history_result.file_count,
                    evaluation_outcome=evaluation_result.outcome,
                    evaluation_row_count=evaluation_result.validation.row_count,
                    evaluation_file_count=evaluation_result.file_count,
                ),
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

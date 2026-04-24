"""Canonical block corpus acquisition workflow."""

from __future__ import annotations

import asyncio
import math
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

from ..acquisition.rpc import BlockRpcClient, RpcController, TimestampRange, evaluation_range
from ..config.models import AcquireConfig
from ..core.files import promote_paths_atomic, prune_empty_directories
from ..core.reporting import Reporter
from ..corpus.builders import ensure_evaluation_dataset, ensure_history_dataset
from ..corpus.io import load_block_frame
from ..corpus.metadata import (
    build_acquire_run_record,
    build_dataset_manifest,
    provider_metadata,
)
from ..corpus.summary import acquire_dry_run_fields, acquisition_result_fields
from ..features import CompiledFeatureContract, compile_feature_contract
from ..storage.corpus import write_dataset_state
from ..storage.layout import resolve_workflow_paths
from ..storage.roots import reindex_root
from ..temporal.contracts import CompiledProblemContract, compile_problem_contract

HISTORY_WINDOW_CUSHION_RATIO = 0.10
HISTORY_REFILL_CUSHION_RATIO = 0.10


def _workflow_facts(config: AcquireConfig) -> list[tuple[str, str]]:
    return [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("provider", config.rpc_endpoint.provider_name),
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


def _with_cushion(value: float, ratio: float) -> int:
    return max(1, math.ceil(value * (1.0 + ratio)))


def _history_window(config: AcquireConfig, window_seconds: int) -> TimestampRange:
    return TimestampRange(
        start=max(0, config.history_window_end_timestamp - window_seconds),
        end=config.history_window_end_timestamp,
    )


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
                cast(Any, self)._initializer,
                cast(Any, self)._initargs,
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
            pending_task for pending_task in asyncio.all_tasks(loop) if not pending_task.done()
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
    paths = resolve_workflow_paths(config)
    history_dir = paths.history_dir
    evaluation_dir = paths.evaluation_dir
    state_db_path = paths.corpus_state_db
    rpc_controller = RpcController.from_config(config.acquisition)
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    evaluation_window = evaluation_range(
        config.evaluation_window_start_timestamp,
        config.evaluation_window_end_timestamp,
    )

    active_reporter = reporter or Reporter()
    active_reporter.header("acquire", _workflow_facts(config))

    block_client = BlockRpcClient(config.rpc_endpoint, config.chain)
    try:
        evaluation_plan = await block_client.plan_window(
            evaluation_window,
            chunk_size=config.acquisition.chunk_size,
        )
        estimated_block_interval_seconds = await block_client.estimate_recent_block_interval()
        bootstrap_history_window_seconds = contract.initial_history_window_seconds(
            estimated_block_interval_seconds,
        )
        requested_history_window_seconds = _with_cushion(
            bootstrap_history_window_seconds,
            HISTORY_WINDOW_CUSHION_RATIO,
        )
        history_plan = await block_client.plan_window(
            _history_window(config, requested_history_window_seconds),
            chunk_size=config.acquisition.chunk_size,
        )

        if config.acquisition.dry_run:
            active_reporter.result(
                "acquire",
                acquire_dry_run_fields(
                    config,
                    contract=contract,
                    history_window_seconds=requested_history_window_seconds,
                    history_plan=history_plan,
                    evaluation_plan=evaluation_plan,
                ),
                status="dry_run",
            )
            return

        current_provider = provider_metadata(config)
        paths.corpus_root.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=paths.corpus_root.parent,
            prefix=f".{paths.corpus_id}.acquire.",
        ) as temp_root_name:
            temp_root = Path(temp_root_name)

            async def _ensure_counted_history(output_dir: Path, working_dir: Path):
                result = await ensure_history_dataset(
                    config=config,
                    block_client=block_client,
                    output_dir=output_dir,
                    working_dir=working_dir,
                    history_plan=history_plan,
                    rpc_controller=rpc_controller,
                    status=active_reporter.milestone,
                )
                return result, _count_valid_history_samples(
                    history_dir=result.path,
                    feature_contract=feature_contract,
                    contract=contract,
                )

            history_result, resolved_capability_samples = await _ensure_counted_history(
                history_dir,
                temp_root / "history-initial",
            )
            if resolved_capability_samples < contract.sample_count:
                validation = history_result.validation
                if (
                    validation.first_timestamp is None
                    or validation.last_timestamp is None
                    or validation.row_count <= 1
                ):
                    raise RuntimeError(
                        "Cannot compute observed history cadence from validation report"
                    )
                sample_shortfall = contract.sample_count - resolved_capability_samples
                observed_seconds_per_block = max(
                    1.0,
                    (validation.last_timestamp - validation.first_timestamp)
                    / (validation.row_count - 1),
                )
                requested_history_window_seconds = max(
                    requested_history_window_seconds,
                    config.history_window_end_timestamp - validation.first_timestamp,
                ) + _with_cushion(
                    sample_shortfall * observed_seconds_per_block,
                    HISTORY_REFILL_CUSHION_RATIO,
                )
                history_plan = await block_client.plan_window(
                    _history_window(config, requested_history_window_seconds),
                    chunk_size=config.acquisition.chunk_size,
                )
                active_reporter.milestone(
                    "history refilling "
                    f"samples={resolved_capability_samples}/{contract.sample_count}"
                )
                history_result, resolved_capability_samples = await _ensure_counted_history(
                    history_result.path,
                    temp_root / "history-refill",
                )
            if resolved_capability_samples < contract.sample_count:
                raise RuntimeError(
                    "History sizing policy under-requested capability samples: "
                    f"valid={resolved_capability_samples}, "
                    f"required={contract.sample_count}"
                )

            evaluation_result = await ensure_evaluation_dataset(
                config=config,
                block_client=block_client,
                output_dir=evaluation_dir,
                working_dir=temp_root,
                evaluation_plan=evaluation_plan,
                rpc_controller=rpc_controller,
                status=active_reporter.milestone,
            )
            manifest = build_dataset_manifest(
                config=config,
                history_request_start_timestamp=history_plan.window.start,
                history_request_end_timestamp=history_plan.window.end,
                evaluation_request_start_timestamp=evaluation_window.start,
                evaluation_request_end_timestamp=evaluation_window.end,
                history_validation=history_result.validation,
                evaluation_validation=evaluation_result.validation,
            )
            acquire_run = build_acquire_run_record(
                config=config,
                provider=current_provider,
                acquisition_runtime=rpc_controller.snapshot(),
                requested_history_window_seconds=requested_history_window_seconds,
                resolved_capability_samples=resolved_capability_samples,
            )
            temp_state_db = temp_root / ".spice" / "state.sqlite"
            write_dataset_state(
                temp_state_db,
                manifest=manifest,
                acquire_run=acquire_run,
            )
            promotions: list[tuple[Path, Path]] = []
            if history_result.promote_dir is not None:
                promotions.append((history_dir, history_result.promote_dir))
            if evaluation_result.promote_dir is not None:
                promotions.append((evaluation_dir, evaluation_result.promote_dir))
            promotions.append((state_db_path, temp_state_db))
            promote_paths_atomic(promotions)
            reindex_root(paths.output_root, root_path=paths.corpus_root)
        active_reporter.result(
            "acquire",
            acquisition_result_fields(
                history_outcome=history_result.outcome,
                history_row_count=history_result.validation.row_count,
                evaluation_outcome=evaluation_result.outcome,
                evaluation_row_count=evaluation_result.validation.row_count,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        prune_empty_directories(
            paths.corpus_root,
            stop_at=paths.corpus_root.parent.parent,
        )
        active_reporter.milestone(
            "acquire cancelled; partial download removed",
            level="warning",
        )
        raise
    except Exception:
        active_reporter.milestone(
            "acquire failed; partial download removed",
            level="warning",
        )
        raise
    finally:
        await block_client.close()

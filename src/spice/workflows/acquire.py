"""Canonical block corpus acquisition workflow."""

from __future__ import annotations

import asyncio

from ..acquisition.rpc import BlockRpcClient
from ..config.models import AcquireConfig
from ..core.async_runtime import run_interruptibly
from ..core.reporting import Reporter
from ..corpus.assembly import (
    CorpusAssemblyRequest,
    acquisition_source_requirements,
    assemble_corpus,
)
from ..corpus.summary import acquire_dry_run_fields, acquisition_result_fields
from ..storage.workflow_roots import resolve_acquire_producer_roots


def _workflow_facts(config: AcquireConfig) -> list[tuple[str, str]]:
    return [
        ("dataset", config.dataset.name),
        ("chain", config.chain.name),
        ("problem", config.problem.id),
        ("provider", config.rpc_endpoint.provider_name),
    ]


def run(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    try:
        run_interruptibly(_run_async(config, reporter=reporter))
    except KeyboardInterrupt:
        return None


async def _run_async(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    roots = resolve_acquire_producer_roots(config)
    active_reporter = reporter or Reporter()
    active_reporter.header("acquire", _workflow_facts(config))
    source_requirements = acquisition_source_requirements(config)
    block_client = BlockRpcClient(
        config.rpc_endpoint,
        config.chain,
        include_priority_fees=(
            "priority_fee_percentiles" in source_requirements.optional_enrichments
        ),
    )
    try:
        result = await assemble_corpus(
            CorpusAssemblyRequest(config=config, roots=roots),
            block_client,
            status=active_reporter.milestone,
        )
        if result.mode == "dry_run":
            active_reporter.result(
                "acquire",
                acquire_dry_run_fields(
                    config,
                    history_window_seconds=result.requested_history_window_seconds,
                    history_plan=result.history_plan,
                    evaluation_plan=result.evaluation_plan,
                ),
                status="dry_run",
            )
            return
        if (
            result.history_outcome is None
            or result.history_row_count is None
            or result.evaluation_outcome is None
            or result.evaluation_row_count is None
        ):
            raise RuntimeError("Committed corpus assembly result is missing split facts")
        active_reporter.result(
            "acquire",
            acquisition_result_fields(
                history_outcome=result.history_outcome,
                history_row_count=result.history_row_count,
                evaluation_outcome=result.evaluation_outcome,
                evaluation_row_count=result.evaluation_row_count,
            ),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        active_reporter.milestone(
            "acquire cancelled; partial staging preserved for resume",
            level="warning",
        )
        raise
    except Exception:
        active_reporter.milestone(
            "acquire failed; partial staging preserved for resume",
            level="warning",
        )
        raise
    finally:
        await block_client.close()

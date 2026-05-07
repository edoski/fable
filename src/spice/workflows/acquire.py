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
from ..storage.workflow_root_materialization import materialize_acquire_roots
from .reporting import (
    acquire_workflow_facts,
    report_acquire_result,
    report_acquire_staging_warning,
)


def run(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    try:
        run_interruptibly(_run_async(config, reporter=reporter))
    except KeyboardInterrupt:
        return None


async def _run_async(config: AcquireConfig, *, reporter: Reporter | None = None) -> None:
    roots = materialize_acquire_roots(config)
    active_reporter = reporter or Reporter()
    active_reporter.header("acquire", acquire_workflow_facts(config))
    source_requirements = acquisition_source_requirements(config)
    block_client = BlockRpcClient(
        config.rpc_endpoint,
        config.chain,
        source_requirements,
    )
    try:
        result = await assemble_corpus(
            CorpusAssemblyRequest(config=config, roots=roots),
            block_client,
            status=active_reporter.milestone,
        )
        report_acquire_result(active_reporter, config=config, result=result)
    except (KeyboardInterrupt, asyncio.CancelledError):
        report_acquire_staging_warning(active_reporter, reason="cancelled")
        raise
    except Exception:
        report_acquire_staging_warning(active_reporter, reason="failed")
        raise
    finally:
        await block_client.close()

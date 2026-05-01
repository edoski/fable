"""Corpus Assembly from acquisition output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..acquisition import (
    AcquisitionPullController,
    BlockPullPlan,
    BlockSource,
)
from ..config.models import AcquireConfig
from ..storage.engine import RootKind
from ..storage.root_handles import AcquireWorkflowRoots
from .acquisition_stage import CorpusAcquisitionStage
from .metadata import (
    AcquireRunRecord,
    DatasetManifest,
    build_acquire_run_record,
    build_dataset_manifest,
    provider_metadata,
)
from .planning import (
    CorpusCapabilityPlanningSpec,
    build_corpus_capability_planning_context,
)
from .split_materialization import (
    CorpusSplitMaterializationSpec,
    CorpusSplitOutcome,
)

StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CorpusAssemblyRequest:
    config: AcquireConfig
    roots: AcquireWorkflowRoots


@dataclass(frozen=True, slots=True)
class CorpusAssemblyResult:
    mode: Literal["dry_run", "committed"]
    history_plan: BlockPullPlan
    evaluation_plan: BlockPullPlan
    requested_history_window_seconds: int
    resolved_capability_samples: int | None
    history_outcome: CorpusSplitOutcome | None
    history_row_count: int | None
    evaluation_outcome: CorpusSplitOutcome | None
    evaluation_row_count: int | None
    manifest: DatasetManifest | None
    acquire_run: AcquireRunRecord | None
    committed_root_kind: RootKind | None


def _noop_status(message: str) -> None:
    del message


def _planning_spec(config: AcquireConfig) -> CorpusCapabilityPlanningSpec:
    return CorpusCapabilityPlanningSpec(
        features=config.features,
        problem=config.problem,
        chain_runtime=config.chain.runtime,
        history_window_end_timestamp=config.history_window_end_timestamp,
        evaluation_window_start_timestamp=config.evaluation_window_start_timestamp,
        evaluation_window_end_timestamp=config.evaluation_window_end_timestamp,
    )


def _split_materialization_spec(config: AcquireConfig) -> CorpusSplitMaterializationSpec:
    return CorpusSplitMaterializationSpec(
        chain_name=config.chain.name,
        expected_chain_id=config.chain.runtime.chain_id,
        chunk_size=config.acquisition.chunk_size,
    )


async def assemble_corpus(
    request: CorpusAssemblyRequest,
    block_source: BlockSource,
    *,
    status: StatusCallback | None = None,
) -> CorpusAssemblyResult:
    config = request.config
    roots = request.roots
    emit = status or _noop_status
    planning_context = build_corpus_capability_planning_context(_planning_spec(config))
    materialization = _split_materialization_spec(config)
    initial_plan = await planning_context.initial_plan(block_source)
    history_plan = initial_plan.history_plan
    evaluation_plan = initial_plan.evaluation_plan
    requested_history_window_seconds = initial_plan.requested_history_window_seconds

    if config.acquisition.dry_run:
        return CorpusAssemblyResult(
            mode="dry_run",
            history_plan=history_plan,
            evaluation_plan=evaluation_plan,
            requested_history_window_seconds=requested_history_window_seconds,
            resolved_capability_samples=None,
            history_outcome=None,
            history_row_count=None,
            evaluation_outcome=None,
            evaluation_row_count=None,
            manifest=None,
            acquire_run=None,
            committed_root_kind=None,
        )

    controller = AcquisitionPullController.from_config(config.acquisition)
    current_provider = provider_metadata(config)
    stage = CorpusAcquisitionStage.open(
        config=config,
        roots=roots,
        planning_context=planning_context,
        materialization=materialization,
        controller=controller,
    )
    fulfillment = await stage.fulfill(
        block_source=block_source,
        initial_history_plan=history_plan,
        evaluation_plan=evaluation_plan,
        requested_history_window_seconds=requested_history_window_seconds,
        status=emit,
    )
    history_plan = fulfillment.history_plan
    evaluation_plan = fulfillment.evaluation_plan
    manifest = build_dataset_manifest(
        config=config,
        dataset_id=roots.corpus.dataset_id,
        history_request_start_timestamp=history_plan.window.start,
        history_request_end_timestamp=history_plan.window.end,
        evaluation_request_start_timestamp=evaluation_plan.window.start,
        evaluation_request_end_timestamp=evaluation_plan.window.end,
        history_validation=fulfillment.history_result.validation,
        evaluation_validation=fulfillment.evaluation_result.validation,
    )
    acquire_run = build_acquire_run_record(
        config=config,
        provider=current_provider,
        acquisition_runtime=controller.snapshot(),
        requested_history_window_seconds=fulfillment.requested_history_window_seconds,
        resolved_capability_samples=fulfillment.resolved_capability_samples,
    )
    committed_root_kind = stage.commit(
        manifest=manifest,
        acquire_run=acquire_run,
        fulfillment=fulfillment,
    )
    return CorpusAssemblyResult(
        mode="committed",
        history_plan=history_plan,
        evaluation_plan=evaluation_plan,
        requested_history_window_seconds=fulfillment.requested_history_window_seconds,
        resolved_capability_samples=fulfillment.resolved_capability_samples,
        history_outcome=fulfillment.history_result.outcome,
        history_row_count=fulfillment.history_result.validation.row_count,
        evaluation_outcome=fulfillment.evaluation_result.outcome,
        evaluation_row_count=fulfillment.evaluation_result.validation.row_count,
        manifest=manifest,
        acquire_run=acquire_run,
        committed_root_kind=committed_root_kind,
    )

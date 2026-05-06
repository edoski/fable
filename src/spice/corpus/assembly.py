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
from ..storage.workflow_roots import AcquireWorkflowRoots
from .acquisition_stage import CorpusAcquisitionStage
from .metadata import (
    AcquireRunRecord,
    DatasetManifest,
)
from .planning import (
    CorpusAcquisitionSourceRequirements,
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


def acquisition_source_requirements(
    config: AcquireConfig,
) -> CorpusAcquisitionSourceRequirements:
    return build_corpus_capability_planning_context(_planning_spec(config)).source_requirements


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
    publication = stage.publish(fulfillment=fulfillment)
    return CorpusAssemblyResult(
        mode="committed",
        history_plan=publication.history_plan,
        evaluation_plan=publication.evaluation_plan,
        requested_history_window_seconds=publication.requested_history_window_seconds,
        resolved_capability_samples=publication.resolved_capability_samples,
        history_outcome=publication.history_outcome,
        history_row_count=publication.history_row_count,
        evaluation_outcome=publication.evaluation_outcome,
        evaluation_row_count=publication.evaluation_row_count,
        manifest=publication.manifest,
        acquire_run=publication.acquire_run,
        committed_root_kind=publication.committed_root_kind,
    )

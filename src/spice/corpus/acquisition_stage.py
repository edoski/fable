"""Corpus acquisition staging and commit."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..acquisition import AcquisitionPullController, BlockPullPlan, BlockSource
from ..config.models import AcquireConfig
from ..core.files import remove_path
from ..storage.corpus import write_dataset_state
from ..storage.engine import RootKind
from ..storage.lifecycle import PartialRootCommit
from ..storage.root_handles import AcquireWorkflowRoots
from .metadata import AcquireRunRecord, DatasetManifest
from .planning import HISTORY_REFILL_ATTEMPT_LIMIT, CorpusCapabilityPlanningContext
from .split_materialization import (
    CorpusSplitMaterializationSpec,
    DatasetBuildResult,
    ensure_evaluation_split,
    ensure_history_split,
)

ACQUIRE_STAGE_DIR_NAME = ".acquire-staging"

StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CorpusAcquisitionStageFulfillment:
    history_result: DatasetBuildResult
    evaluation_result: DatasetBuildResult
    history_plan: BlockPullPlan
    evaluation_plan: BlockPullPlan
    requested_history_window_seconds: int
    resolved_capability_samples: int


@dataclass(frozen=True, slots=True)
class CorpusAcquisitionStage:
    config: AcquireConfig
    roots: AcquireWorkflowRoots
    planning_context: CorpusCapabilityPlanningContext
    materialization: CorpusSplitMaterializationSpec
    controller: AcquisitionPullController
    temp_root: Path

    @classmethod
    def open(
        cls,
        *,
        config: AcquireConfig,
        roots: AcquireWorkflowRoots,
        planning_context: CorpusCapabilityPlanningContext,
        materialization: CorpusSplitMaterializationSpec,
        controller: AcquisitionPullController,
    ) -> CorpusAcquisitionStage:
        roots.corpus.root_path.parent.mkdir(parents=True, exist_ok=True)
        temp_root = acquire_stage_root(roots)
        temp_root.mkdir(parents=True, exist_ok=True)
        write_acquire_stage_record(
            temp_root / ".spice" / "acquire-stage.json",
            config=config,
            corpus_id=roots.corpus.dataset_id,
        )
        return cls(
            config=config,
            roots=roots,
            planning_context=planning_context,
            materialization=materialization,
            controller=controller,
            temp_root=temp_root,
        )

    async def fulfill(
        self,
        *,
        block_source: BlockSource,
        initial_history_plan: BlockPullPlan,
        evaluation_plan: BlockPullPlan,
        requested_history_window_seconds: int,
        status: StatusCallback,
    ) -> CorpusAcquisitionStageFulfillment:
        history_result, resolved_capability_samples, history_plan, resolved_history_seconds = (
            await self._ensure_sufficient_history(
                block_source=block_source,
                initial_history_plan=initial_history_plan,
                requested_history_window_seconds=requested_history_window_seconds,
                status=status,
            )
        )
        evaluation_result = await ensure_evaluation_split(
            materialization=self.materialization,
            block_source=block_source,
            output_dir=self.roots.corpus.evaluation_dir,
            working_dir=self.temp_root,
            evaluation_plan=evaluation_plan,
            controller=self.controller,
            status=status,
        )
        return CorpusAcquisitionStageFulfillment(
            history_result=history_result,
            evaluation_result=evaluation_result,
            history_plan=history_plan,
            evaluation_plan=evaluation_plan,
            requested_history_window_seconds=resolved_history_seconds,
            resolved_capability_samples=resolved_capability_samples,
        )

    def commit(
        self,
        *,
        manifest: DatasetManifest,
        acquire_run: AcquireRunRecord,
        fulfillment: CorpusAcquisitionStageFulfillment,
    ) -> RootKind:
        temp_state_db = self.temp_root / ".spice" / "state.sqlite"
        write_dataset_state(
            temp_state_db,
            manifest=manifest,
            acquire_run=acquire_run,
        )
        commit = PartialRootCommit(
            storage_root=self.roots.storage.root_path,
            root_path=self.roots.corpus.root_path,
        )
        commit.add(self.roots.corpus.history_dir, fulfillment.history_result.promote_dir)
        commit.add(self.roots.corpus.evaluation_dir, fulfillment.evaluation_result.promote_dir)
        commit.add(self.roots.corpus.state_db_path, temp_state_db)
        committed_root_kind = commit.commit().root_kind
        remove_path(self.temp_root)
        return committed_root_kind

    async def _ensure_sufficient_history(
        self,
        *,
        block_source: BlockSource,
        initial_history_plan: BlockPullPlan,
        requested_history_window_seconds: int,
        status: StatusCallback,
    ) -> tuple[DatasetBuildResult, int, BlockPullPlan, int]:
        history_plan = initial_history_plan
        history_result = await ensure_history_split(
            materialization=self.materialization,
            block_source=block_source,
            output_dir=self.roots.corpus.history_dir,
            working_dir=self.temp_root / "history-initial",
            history_plan=history_plan,
            controller=self.controller,
            status=status,
        )
        resolved_capability_samples = self.planning_context.count_valid_history_samples(
            history_result.path,
        )

        for refill_attempt in range(1, HISTORY_REFILL_ATTEMPT_LIMIT + 1):
            refill_plan = await self.planning_context.plan_history_refill(
                block_source=block_source,
                validation=history_result.validation,
                resolved_capability_samples=resolved_capability_samples,
                requested_history_window_seconds=requested_history_window_seconds,
            )
            if refill_plan is None:
                break
            requested_history_window_seconds = refill_plan.requested_history_window_seconds
            history_plan = refill_plan.history_plan
            status(refill_plan.status_message)
            history_result = await ensure_history_split(
                materialization=self.materialization,
                block_source=block_source,
                output_dir=history_result.path,
                working_dir=self.temp_root / f"history-refill-{refill_attempt}",
                history_plan=history_plan,
                controller=self.controller,
                status=status,
            )
            resolved_capability_samples = self.planning_context.count_valid_history_samples(
                history_result.path,
            )

        self.planning_context.ensure_sufficient_history_samples(resolved_capability_samples)
        return (
            history_result,
            resolved_capability_samples,
            history_plan,
            requested_history_window_seconds,
        )


def acquire_stage_root(roots: AcquireWorkflowRoots) -> Path:
    return roots.corpus.root_path.parent / f".{roots.corpus.dataset_id}{ACQUIRE_STAGE_DIR_NAME}"


def write_acquire_stage_record(
    path: Path,
    *,
    config: AcquireConfig,
    corpus_id: str,
) -> None:
    record = {
        "chain": config.chain.name,
        "chain_id": config.chain.runtime.chain_id,
        "dataset": config.dataset.name,
        "evaluation_date": config.dataset.evaluation_date.isoformat(),
        "corpus_id": corpus_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")

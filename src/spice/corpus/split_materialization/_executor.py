"""Corpus split executable materialization plan runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, assert_never

from ...acquisition import AcquisitionPullController, BlockPullPlan, BlockSource
from ._dataset_state import reused_result, staged_result
from ._policy import (
    CorpusSplitOutcome,
    ExtendEvaluationCommittedSplitPlan,
    ExtendHistoryCommittedSplitPlan,
    MaterializeFullSplitPlan,
    RejectInvalidStagedSplitPlan,
    ReuseCommittedSplitPlan,
    ReuseStagedSplitPlan,
    SplitMaterializationPlan,
)
from ._pulls import plan_pull_dir, pull_plan_range_to_dir, pull_plan_to_frame
from ._reuse import (
    materialize_dataset,
    materialize_dataset_from_sources,
    reusable_block_files_and_edges,
)
from ._types import (
    CorpusSplitKind,
    CorpusSplitMaterializationSpec,
    DatasetBuildResult,
    StatusCallback,
    ValidationCallback,
)


def reject_invalid_staged(plan: RejectInvalidStagedSplitPlan) -> NoReturn:
    raise RuntimeError(f"{plan.error_message}: {plan.dataset.validation}")


@dataclass(frozen=True, slots=True)
class SplitPlanExecution:
    kind: CorpusSplitKind
    plan: BlockPullPlan
    working_dir: Path
    materialization: CorpusSplitMaterializationSpec
    block_source: BlockSource
    controller: AcquisitionPullController
    emit: StatusCallback
    validate_result: ValidationCallback


async def execute_split_materialization_plan(
    plan: SplitMaterializationPlan,
    *,
    execution: SplitPlanExecution,
) -> DatasetBuildResult:
    if isinstance(plan, RejectInvalidStagedSplitPlan):
        reject_invalid_staged(plan)

    if isinstance(plan, ReuseStagedSplitPlan):
        execution.emit(plan.status_message)
        return staged_result(
            plan.dataset,
            validation=plan.validation,
            outcome=plan.outcome,
        )

    if isinstance(plan, ReuseCommittedSplitPlan):
        execution.emit(plan.status_message)
        return reused_result(plan.dataset, validation=plan.validation)

    if isinstance(plan, ExtendHistoryCommittedSplitPlan):
        return await _extend_history_committed_split(plan, execution=execution)

    if isinstance(plan, ExtendEvaluationCommittedSplitPlan):
        return await _extend_evaluation_committed_split(plan, execution=execution)

    if isinstance(plan, MaterializeFullSplitPlan):
        return await _materialize_full_split(plan, execution=execution)

    assert_never(plan)


async def _extend_history_committed_split(
    plan: ExtendHistoryCommittedSplitPlan,
    *,
    execution: SplitPlanExecution,
) -> DatasetBuildResult:
    execution.emit(plan.status_message)
    prefix_dir = await pull_plan_range_to_dir(
        block_source=execution.block_source,
        pull_range=plan.prefix,
        window=execution.plan.window,
        working_dir=execution.working_dir,
        materialization=execution.materialization,
        controller=execution.controller,
    )
    return materialize_dataset_from_sources(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        source_dirs=(prefix_dir, plan.existing.path),
        outcome=CorpusSplitOutcome.EXTENDED,
    )


async def _extend_evaluation_committed_split(
    plan: ExtendEvaluationCommittedSplitPlan,
    *,
    execution: SplitPlanExecution,
) -> DatasetBuildResult:
    source_dirs: list[Path] = []
    source_files, frames = reusable_block_files_and_edges(
        plan.existing.path,
        block_range=plan.reusable_range,
    )
    for pull_range in plan.pull_ranges:
        source_dirs.append(
            await pull_plan_range_to_dir(
                block_source=execution.block_source,
                pull_range=pull_range,
                window=execution.plan.window,
                working_dir=execution.working_dir,
                materialization=execution.materialization,
                controller=execution.controller,
            )
        )
    execution.emit(plan.status_message)
    return materialize_dataset_from_sources(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        source_dirs=source_dirs,
        source_files=source_files,
        frames=frames,
        outcome=CorpusSplitOutcome.EXTENDED,
    )


async def _materialize_full_split(
    plan: MaterializeFullSplitPlan,
    *,
    execution: SplitPlanExecution,
) -> DatasetBuildResult:
    execution.emit(plan.status_message)
    frame = await pull_plan_to_frame(
        block_source=execution.block_source,
        plan=execution.plan,
        output_dir=plan_pull_dir(
            execution.working_dir,
            label=execution.kind.value,
            plan=execution.plan,
        ),
        materialization=execution.materialization,
        controller=execution.controller,
    )
    return materialize_dataset(
        mode=execution.kind.value,
        materialization=execution.materialization,
        working_dir=execution.working_dir,
        validate_result=execution.validate_result,
        frames=(frame,),
        outcome=plan.outcome,
    )

"""Corpus split materialization session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...acquisition import AcquisitionPullController, BlockSource
from ..validation import BlockDatasetValidationReport
from ._dataset_state import load_existing_dataset, split_candidate
from ._executor import SplitPlanExecution, execute_split_materialization_plan
from ._invariants import (
    reusable_range_matches_target_window,
    validate_evaluation_result,
    validate_history_result,
)
from ._policy import (
    SplitTarget,
    plan_evaluation_split_materialization,
    plan_history_split_materialization,
)
from ._types import (
    CorpusSplitIntent,
    CorpusSplitKind,
    CorpusSplitMaterializationSpec,
    DatasetBuildResult,
    StatusCallback,
)


@dataclass(frozen=True, slots=True)
class CorpusSplitMaterializationSession:
    materialization: CorpusSplitMaterializationSpec
    block_source: BlockSource
    controller: AcquisitionPullController
    status: StatusCallback | None = None

    async def fulfill(self, intent: CorpusSplitIntent) -> DatasetBuildResult:
        if intent.kind is CorpusSplitKind.HISTORY:
            return await _ensure_history_split(
                intent,
                materialization=self.materialization,
                block_source=self.block_source,
                controller=self.controller,
                status=self.status,
            )
        if intent.kind is CorpusSplitKind.EVALUATION:
            return await _ensure_evaluation_split(
                intent,
                materialization=self.materialization,
                block_source=self.block_source,
                controller=self.controller,
                status=self.status,
            )
        raise ValueError(f"Unsupported corpus split kind: {intent.kind}")


def noop_status(message: str) -> None:
    del message


async def _ensure_history_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> DatasetBuildResult:
    history_plan = intent.plan
    emit = status or noop_status
    existing = load_existing_dataset(
        intent.output_dir,
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )
    staged = load_existing_dataset(
        intent.working_dir / "history",
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )

    def validate_result(validation: BlockDatasetValidationReport, _: Path) -> None:
        validate_history_result(validation, history_plan=history_plan)

    materialization_plan = plan_history_split_materialization(
        SplitTarget(
            kind=intent.kind.value,
            block_range=history_plan.block_range,
            window=history_plan.window,
        ),
        existing=split_candidate(existing),
        staged=split_candidate(staged),
        validate_target=validate_result,
    )

    return await execute_split_materialization_plan(
        materialization_plan,
        execution=SplitPlanExecution(
            kind=intent.kind,
            plan=history_plan,
            working_dir=intent.working_dir,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            emit=emit,
            validate_result=validate_result,
        ),
    )


async def _ensure_evaluation_split(
    intent: CorpusSplitIntent,
    *,
    materialization: CorpusSplitMaterializationSpec,
    block_source: BlockSource,
    controller: AcquisitionPullController,
    status: StatusCallback | None,
) -> DatasetBuildResult:
    evaluation_plan = intent.plan
    emit = status or noop_status
    existing = load_existing_dataset(
        intent.output_dir,
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )
    staged = load_existing_dataset(
        intent.working_dir / "evaluation",
        expected_chain_id=materialization.expected_chain_id,
        required_columns=materialization.required_columns,
    )

    def validate_result(validation: BlockDatasetValidationReport, dataset_dir: Path) -> None:
        validate_evaluation_result(
            validation,
            evaluation_dir=dataset_dir,
            evaluation_plan=evaluation_plan,
            expected_chain_id=materialization.expected_chain_id,
            required_columns=materialization.required_columns,
        )

    materialization_plan = plan_evaluation_split_materialization(
        SplitTarget(
            kind=intent.kind.value,
            block_range=evaluation_plan.block_range,
            window=evaluation_plan.window,
        ),
        existing=split_candidate(existing),
        staged=split_candidate(staged),
        validate_target=validate_result,
        reusable_range_matches_target_window=reusable_range_matches_target_window,
    )

    return await execute_split_materialization_plan(
        materialization_plan,
        execution=SplitPlanExecution(
            kind=intent.kind,
            plan=evaluation_plan,
            working_dir=intent.working_dir,
            materialization=materialization,
            block_source=block_source,
            controller=controller,
            emit=emit,
            validate_result=validate_result,
        ),
    )

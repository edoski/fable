"""History dataset builder."""

from __future__ import annotations

from pathlib import Path

from ...acquisition.rpc import BlockPullPlan, BlockRpcClient, RpcController
from ...config import AcquireConfig
from ...core.reporting import Reporter
from .shared import (
    DatasetBuildOutcome,
    DatasetBuildResult,
    StageUpdateCallback,
    block_range_end,
    block_range_start,
    load_existing_dataset,
    materialize_dataset,
    noop_stage_update,
    partial_plan,
    pull_plan_to_frame,
    reused_result,
    validate_history_result,
)


async def ensure_history_dataset(
    *,
    config: AcquireConfig,
    block_client: BlockRpcClient,
    output_dir: Path,
    working_dir: Path,
    history_plan: BlockPullPlan,
    rpc_controller: RpcController,
    reporter: Reporter,
    stage_update: StageUpdateCallback | None = None,
) -> DatasetBuildResult:
    update_stage = stage_update or noop_stage_update
    expected_chain_id = config.chain.runtime.chain_id
    existing = load_existing_dataset(output_dir, expected_chain_id=expected_chain_id)

    def validate_result(validation, _: Path) -> None:
        validate_history_result(validation, history_plan=history_plan)

    update_stage("planning", "checking existing dataset")
    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean history validation requires an in-memory frame")
        existing_end = block_range_end(existing.validation)
        if existing_end == history_plan.block_range.end:
            existing_start = block_range_start(existing.validation)
            if existing_start <= history_plan.block_range.start:
                update_stage("planning", "validating cached dataset")
                validate_history_result(existing.validation, history_plan=history_plan)
                return reused_result(existing)

            prefix_plan = partial_plan(
                block_client,
                start_block=history_plan.block_range.start,
                end_block=existing_start,
                window=history_plan.window,
                chunk_size=config.acquisition.chunk_size,
            )
            if prefix_plan is None:
                raise RuntimeError("history prefix plan unexpectedly resolved to empty")
            update_stage("planning", "extending cached dataset")
            prefix_frame = await pull_plan_to_frame(
                block_client=block_client,
                plan=prefix_plan,
                output_dir=working_dir / "history-prefix",
                chunk_size=config.acquisition.chunk_size,
                rpc_controller=rpc_controller,
                reporter=reporter,
                progress_total=history_plan.expected_rows,
                progress_completed=existing.validation.row_count,
            )
            return materialize_dataset(
                mode="history",
                config=config,
                working_dir=working_dir,
                expected_chain_id=expected_chain_id,
                stage_update=update_stage,
                validate_result=validate_result,
                frames=(prefix_frame, existing.frame),
                outcome=DatasetBuildOutcome.EXTENDED,
            )

    update_stage("planning", "preparing download")
    await pull_plan_to_frame(
        block_client=block_client,
        plan=history_plan,
        output_dir=working_dir / "history",
        chunk_size=config.acquisition.chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
        progress_total=history_plan.expected_rows,
    )
    return materialize_dataset(
        mode="history",
        config=config,
        working_dir=working_dir,
        expected_chain_id=expected_chain_id,
        stage_update=update_stage,
        validate_result=validate_result,
        outcome=(
            DatasetBuildOutcome.REBUILT
            if existing is not None
            else DatasetBuildOutcome.CREATED
        ),
    )

"""History dataset builder."""

from __future__ import annotations

from pathlib import Path

from ...acquisition.rpc import BlockPullPlan, RpcController, Web3BlockClient
from ...config import AcquireConfig
from ...core.reporting import Reporter
from ...corpus.io import iter_block_files
from .shared import (
    DatasetBuildOutcome,
    DatasetBuildResult,
    StageUpdateCallback,
    block_range_end,
    block_range_start,
    combined_frame,
    load_existing_dataset,
    noop_stage_update,
    partial_plan,
    pull_plan_to_frame,
    validate_block_dataset,
    validate_history_result,
    write_block_dataset_dir,
)


async def ensure_history_dataset(
    *,
    config: AcquireConfig,
    block_client: Web3BlockClient,
    output_dir: Path,
    working_dir: Path,
    history_plan: BlockPullPlan,
    rpc_controller: RpcController,
    reporter: Reporter,
    stage_update: StageUpdateCallback | None = None,
) -> DatasetBuildResult:
    chunk_size = config.acquisition.chunk_size
    update_stage = stage_update or noop_stage_update
    update_stage("planning", "checking existing dataset")
    existing = load_existing_dataset(output_dir, expected_chain_id=config.chain.runtime.chain_id)
    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean history validation requires an in-memory frame")
        existing_end = block_range_end(existing.validation)
        if existing_end == history_plan.block_range.end:
            existing_start = block_range_start(existing.validation)
            if existing_start <= history_plan.block_range.start:
                update_stage("planning", "validating cached dataset")
                validate_history_result(
                    existing.validation,
                    history_plan=history_plan,
                )
                return DatasetBuildResult(
                    path=output_dir,
                    validation=existing.validation,
                    file_count=existing.file_count,
                    promote_dir=None,
                    pulled_blocks=False,
                    outcome=DatasetBuildOutcome.REUSED,
                )

            prefix_plan = partial_plan(
                block_client,
                start_block=history_plan.block_range.start,
                end_block=existing_start,
                window=history_plan.window,
                chunk_size=chunk_size,
            )
            if prefix_plan is None:
                raise RuntimeError("history prefix plan unexpectedly resolved to empty")
            update_stage("planning", "extending cached dataset")
            prefix_frame = await pull_plan_to_frame(
                block_client=block_client,
                plan=prefix_plan,
                output_dir=working_dir / "history-prefix",
                chunk_size=chunk_size,
                rpc_controller=rpc_controller,
                reporter=reporter,
                progress_total=history_plan.expected_rows,
                progress_completed=existing.validation.row_count,
            )
            update_stage("planning", "writing merged dataset")
            history_frame = combined_frame(prefix_frame, existing.frame)
            file_count = write_block_dataset_dir(
                working_dir / "history",
                frame=history_frame,
                chunk_size=chunk_size,
                chain_name=config.chain.name,
            )
            update_stage("planning", "validating dataset")
            validation = validate_block_dataset(
                working_dir / "history",
                expected_chain_id=config.chain.runtime.chain_id,
            )
            validate_history_result(
                validation,
                history_plan=history_plan,
            )
            return DatasetBuildResult(
                path=working_dir / "history",
                validation=validation,
                file_count=file_count,
                promote_dir=working_dir / "history",
                pulled_blocks=True,
                outcome=DatasetBuildOutcome.EXTENDED,
            )

    update_stage("planning", "preparing download")
    pulled_frame = await pull_plan_to_frame(
        block_client=block_client,
        plan=history_plan,
        output_dir=working_dir / "history",
        chunk_size=chunk_size,
        rpc_controller=rpc_controller,
        reporter=reporter,
        progress_total=history_plan.expected_rows,
    )
    update_stage("planning", "validating dataset")
    validation = validate_block_dataset(
        working_dir / "history",
        expected_chain_id=config.chain.runtime.chain_id,
    )
    validate_history_result(
        validation,
        history_plan=history_plan,
    )
    return DatasetBuildResult(
        path=working_dir / "history",
        validation=validation,
        file_count=len(iter_block_files(working_dir / "history")),
        promote_dir=working_dir / "history",
        pulled_blocks=pulled_frame.height > 0,
        outcome=(
            DatasetBuildOutcome.REBUILT
            if existing is not None
            else DatasetBuildOutcome.CREATED
        ),
    )

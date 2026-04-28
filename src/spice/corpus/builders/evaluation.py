"""Evaluation dataset builder."""

from __future__ import annotations

from pathlib import Path

from ...acquisition.rpc import BlockPullPlan, BlockRange, BlockRpcClient, RpcController
from ...config.models import AcquireConfig
from .shared import (
    DatasetBuildOutcome,
    DatasetBuildResult,
    StatusCallback,
    block_range_end,
    block_range_start,
    filter_block_range,
    load_existing_dataset,
    materialize_dataset,
    noop_status,
    partial_plan,
    plan_pull_dir,
    pull_plan_to_frame,
    reused_result,
    staged_result,
    validate_evaluation_result,
)


async def ensure_evaluation_dataset(
    *,
    config: AcquireConfig,
    block_client: BlockRpcClient,
    output_dir: Path,
    working_dir: Path,
    evaluation_plan: BlockPullPlan,
    rpc_controller: RpcController,
    status: StatusCallback | None = None,
) -> DatasetBuildResult:
    emit = status or noop_status
    expected_chain_id = config.chain.runtime.chain_id
    existing = load_existing_dataset(output_dir, expected_chain_id=expected_chain_id)
    staged = load_existing_dataset(
        working_dir / "evaluation",
        expected_chain_id=expected_chain_id,
    )

    def validate_result(validation, dataset_dir: Path) -> None:
        validate_evaluation_result(
            validation,
            evaluation_dir=dataset_dir,
            evaluation_plan=evaluation_plan,
            expected_chain_id=expected_chain_id,
        )

    if staged is not None:
        if staged.validation.status != "clean":
            raise RuntimeError(
                f"Cannot resume invalid staged evaluation dataset: {staged.validation}"
            )
        staged_validation = staged.validation.model_copy(deep=True)
        try:
            validate_result(staged_validation, staged.path)
        except ValueError:
            pass
        else:
            emit("evaluation reused staged dataset")
            return staged_result(
                staged,
                validation=staged_validation,
                outcome=(
                    DatasetBuildOutcome.REBUILT
                    if existing is not None
                    else DatasetBuildOutcome.CREATED
                ),
            )

    if existing is not None and existing.validation.status == "clean":
        if existing.frame is None:
            raise RuntimeError("clean evaluation validation requires an in-memory frame")
        existing_start = block_range_start(existing.validation)
        existing_end = block_range_end(existing.validation)
        target_start = evaluation_plan.block_range.start
        target_end = evaluation_plan.block_range.end

        if existing_start == target_start and existing_end == target_end:
            validation = existing.validation.model_copy(deep=True)
            emit("evaluation reused cached dataset")
            validate_result(validation, existing.path)
            return reused_result(existing, validation=validation)

        overlap_start = max(existing_start, target_start)
        overlap_end = min(existing_end, target_end)
        if overlap_end > overlap_start:
            frames = [
                filter_block_range(existing.frame, BlockRange(start=overlap_start, end=overlap_end))
            ]

            prefix_plan = partial_plan(
                block_client,
                start_block=target_start,
                end_block=overlap_start,
                window=evaluation_plan.window,
            )
            if prefix_plan is not None:
                frames.insert(
                    0,
                    await pull_plan_to_frame(
                        block_client=block_client,
                        plan=prefix_plan,
                        output_dir=plan_pull_dir(
                            working_dir,
                            label="evaluation-prefix",
                            plan=prefix_plan,
                        ),
                        chunk_size=config.acquisition.chunk_size,
                        rpc_controller=rpc_controller,
                    ),
                )

            suffix_plan = partial_plan(
                block_client,
                start_block=overlap_end,
                end_block=target_end,
                window=evaluation_plan.window,
            )
            if suffix_plan is not None:
                frames.append(
                    await pull_plan_to_frame(
                        block_client=block_client,
                        plan=suffix_plan,
                        output_dir=plan_pull_dir(
                            working_dir,
                            label="evaluation-suffix",
                            plan=suffix_plan,
                        ),
                        chunk_size=config.acquisition.chunk_size,
                        rpc_controller=rpc_controller,
                    )
                )

            emit("evaluation extending cached dataset")
            return materialize_dataset(
                mode="evaluation",
                config=config,
                working_dir=working_dir,
                expected_chain_id=expected_chain_id,
                validate_result=validate_result,
                frames=frames,
                outcome=DatasetBuildOutcome.EXTENDED,
            )

    emit("evaluation downloading")
    frame = await pull_plan_to_frame(
        block_client=block_client,
        plan=evaluation_plan,
        output_dir=plan_pull_dir(working_dir, label="evaluation", plan=evaluation_plan),
        chunk_size=config.acquisition.chunk_size,
        rpc_controller=rpc_controller,
    )
    return materialize_dataset(
        mode="evaluation",
        config=config,
        working_dir=working_dir,
        expected_chain_id=expected_chain_id,
        validate_result=validate_result,
        frames=(frame,),
        outcome=(
            DatasetBuildOutcome.REBUILT
            if existing is not None
            else DatasetBuildOutcome.CREATED
        ),
    )

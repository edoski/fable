"""Canonical dataset builders for acquisition."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core.config import ExperimentConfig
from ..core.console import Reporter
from ..data.io import load_block_frame
from ..data.validation import BlockDatasetValidationReport, validate_exact_window_frame
from .metadata import has_block_files
from .rpc import BlockPullPlan, Web3BlockClient

MAX_HISTORY_WINDOW_ATTEMPTS = 3


def validate_block_dataset(
    path: Path,
    *,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> BlockDatasetValidationReport:
    try:
        frame = load_block_frame(path)
    except Exception as exc:  # pragma: no cover - surfaced in workflow smoke tests
        return BlockDatasetValidationReport(
            dataset_path=path,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
            status="error",
            errors=[str(exc)],
        )
    return validate_exact_window_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )


def ensure_block_dataset(
    *,
    block_client: Web3BlockClient,
    output_dir: Path,
    plan: BlockPullPlan,
    expected_chain_id: int,
    chunk_size: int,
    rpc_batch_size: int,
    overwrite: bool,
    reporter: Reporter,
) -> tuple[BlockPullPlan | None, BlockDatasetValidationReport]:
    if not overwrite and has_block_files(output_dir):
        validate_existing_task = reporter.start_task(f"validate dataset {output_dir.name}")
        validation = validate_block_dataset(
            output_dir,
            expected_chain_id=expected_chain_id,
            expected_start_timestamp=plan.window.start,
            expected_end_timestamp=plan.window.end,
        )
        reporter.finish_task(
            validate_existing_task,
            message=f"{output_dir} ({validation.status})",
        )
        if validation.status == "clean":
            reporter.log(f"reusing canonical dataset: {output_dir}")
            return None, validation
        reporter.log(
            f"rebuilding dataset after failed validation: {output_dir}",
            level="warning",
        )

    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    pulled_plan = block_client.pull_block_range(
        output_dir,
        plan=plan,
        chunk_size=chunk_size,
        rpc_batch_size=rpc_batch_size,
        reporter=reporter,
    )
    validate_final_task = reporter.start_task(f"validate dataset {output_dir.name}")
    validation = validate_block_dataset(
        output_dir,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=plan.window.start,
        expected_end_timestamp=plan.window.end,
    )
    reporter.finish_task(validate_final_task, message=f"{output_dir} ({validation.status})")
    if validation.status != "clean":
        raise ValueError(f"Canonical dataset validation failed for {output_dir}: {validation}")
    return pulled_plan, validation


def ensure_history_dataset(
    *,
    config: ExperimentConfig,
    block_client: Web3BlockClient,
    output_dir: Path,
    history_plan: BlockPullPlan,
    required_history_blocks: int,
    reporter: Reporter,
) -> tuple[BlockPullPlan | None, BlockDatasetValidationReport, BlockPullPlan]:
    current_plan = history_plan
    overwrite = config.acquisition.overwrite
    pulled_plan: BlockPullPlan | None = None
    validation: BlockDatasetValidationReport | None = None

    for attempt_index in range(MAX_HISTORY_WINDOW_ATTEMPTS):
        pulled_plan, validation = ensure_block_dataset(
            block_client=block_client,
            output_dir=output_dir,
            plan=current_plan,
            expected_chain_id=config.chain.chain_id,
            chunk_size=config.acquisition.chunk_size,
            rpc_batch_size=config.acquisition.rpc_batch_size,
            overwrite=overwrite,
            reporter=reporter,
        )
        if validation.row_count >= required_history_blocks:
            return pulled_plan, validation, current_plan
        if attempt_index == MAX_HISTORY_WINDOW_ATTEMPTS - 1:
            break

        expanded_plan = block_client.expand_history_plan(
            current_plan,
            observed_row_count=validation.row_count,
            required_history_blocks=required_history_blocks,
            chunk_size=config.acquisition.chunk_size,
        )
        reporter.log(
            "expanding history plan backward "
            f"from block {current_plan.block_range.start} to {expanded_plan.block_range.start} "
            f"for {required_history_blocks} required blocks",
            level="warning",
        )
        current_plan = expanded_plan
        overwrite = True

    if validation is None:
        raise RuntimeError("history acquisition finished without a validation report")
    raise ValueError(
        "History dataset is too short after repeated expansion; "
        f"need at least {required_history_blocks} blocks, "
        f"got {validation.row_count}"
    )


def ensure_evaluation_dataset(
    *,
    config: ExperimentConfig,
    block_client: Web3BlockClient,
    output_dir: Path,
    evaluation_plan: BlockPullPlan,
    reporter: Reporter,
) -> tuple[BlockPullPlan | None, BlockDatasetValidationReport]:
    return ensure_block_dataset(
        block_client=block_client,
        output_dir=output_dir,
        plan=evaluation_plan,
        expected_chain_id=config.chain.chain_id,
        chunk_size=config.acquisition.chunk_size,
        rpc_batch_size=config.acquisition.rpc_batch_size,
        overwrite=config.acquisition.overwrite,
        reporter=reporter,
    )

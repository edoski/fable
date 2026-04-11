"""Canonical raw and enriched dataset builders for acquisition."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.config import ExperimentConfig
from ..core.console import Reporter
from ..data.io import load_enriched_block_frame
from ..data.validation import BlockDatasetValidationReport, validate_exact_window_frame
from .cryo import CryoRunResult, TimestampRange, run_cryo
from .enrich import enrich_path
from .metadata import has_block_files
from .raw_normalization import normalize_raw_dataset
from .raw_validation import RawPullValidationReport, validate_raw_pull
from .windowing import expanded_history_range


def validate_enriched_dataset(
    path: Path,
    *,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
) -> BlockDatasetValidationReport:
    try:
        frame = load_enriched_block_frame(path)
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


def run_raw_pull(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    window: TimestampRange,
    reporter: Reporter,
    overwrite: bool,
    dry_run: bool,
) -> CryoRunResult:
    return run_cryo(
        config.chain,
        config.acquisition,
        output_dir,
        window,
        provider=config.provider,
        overwrite=overwrite,
        dry_run=dry_run,
        reporter=reporter,
    )


def ensure_canonical_raw_dataset(
    *,
    chain_name: str,
    chain_id: int,
    output_dir: Path,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
    chunk_size: int,
    overwrite: bool,
    run_pull: Callable[[Path], CryoRunResult],
    reporter: Reporter,
) -> tuple[CryoRunResult | None, RawPullValidationReport]:
    if not overwrite and has_block_files(output_dir):
        validate_existing_task = reporter.start_task(f"validate raw dataset {output_dir.name}")
        validation = validate_raw_pull(
            output_dir,
            expected_chain_name=chain_name,
            expected_chain_id=chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
            expected_chunk_size=chunk_size,
        )
        reporter.finish_task(
            validate_existing_task,
            message=f"{output_dir} ({validation.status})",
        )
        if validation.status == "clean":
            reporter.log(f"reusing canonical raw dataset: {output_dir}")
            return None, validation
        reporter.log(
            f"rebuilding raw dataset after failed validation: {output_dir}",
            level="warning",
        )

    with TemporaryDirectory(prefix=f"spice-{chain_name}-{output_dir.name}-raw-") as scratch_root:
        scratch_dir = Path(scratch_root) / output_dir.name
        pull_result = run_pull(scratch_dir)
        normalize_task = reporter.start_task(f"normalize raw dataset {output_dir.name}")
        normalize_raw_dataset(
            scratch_dir,
            output_dir,
            chain_name=chain_name,
            expected_chain_id=chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
            chunk_size=chunk_size,
        )
        reporter.finish_task(normalize_task, message=str(output_dir))

    validate_final_task = reporter.start_task(f"validate raw dataset {output_dir.name}")
    validation = validate_raw_pull(
        output_dir,
        expected_chain_name=chain_name,
        expected_chain_id=chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
        expected_chunk_size=chunk_size,
    )
    reporter.finish_task(validate_final_task, message=f"{output_dir} ({validation.status})")
    if validation.status != "clean":
        raise ValueError(f"Canonical raw dataset validation failed for {output_dir}: {validation}")
    return pull_result, validation


def ensure_enriched_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_chain_id: int,
    expected_start_timestamp: int,
    expected_end_timestamp: int,
    overwrite: bool,
    fetch_gas_limits,
    batch_size: int,
    max_methods_per_second: float,
    reporter: Reporter,
) -> BlockDatasetValidationReport:
    if not overwrite and has_block_files(output_dir):
        validate_existing_task = reporter.start_task(
            f"validate enriched dataset {output_dir.name}"
        )
        validation = validate_enriched_dataset(
            output_dir,
            expected_chain_id=expected_chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
        )
        reporter.finish_task(
            validate_existing_task,
            message=f"{output_dir} ({validation.status})",
        )
        if validation.status == "clean":
            reporter.log(f"reusing canonical enriched dataset: {output_dir}")
            return validation
        reporter.log(
            f"rebuilding enriched dataset after failed validation: {output_dir}",
            level="warning",
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    enrich_path(
        input_dir,
        output_dir,
        fetch_gas_limits=fetch_gas_limits,
        batch_size=batch_size,
        max_methods_per_second=max_methods_per_second,
        reporter=reporter,
    )
    validate_final_task = reporter.start_task(f"validate enriched dataset {output_dir.name}")
    validation = validate_enriched_dataset(
        output_dir,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    reporter.finish_task(validate_final_task, message=f"{output_dir} ({validation.status})")
    if validation.status != "clean":
        raise ValueError(
            f"Canonical enriched dataset validation failed for {output_dir}: {validation}"
        )
    return validation


def ensure_history_raw_dataset(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    history_window: TimestampRange,
    required_history_blocks: int,
    reporter: Reporter,
) -> tuple[CryoRunResult | None, RawPullValidationReport, TimestampRange]:
    result, validation = ensure_canonical_raw_dataset(
        chain_name=config.chain.name.value,
        chain_id=config.chain.chain_id,
        output_dir=output_dir,
        expected_start_timestamp=history_window.start,
        expected_end_timestamp=history_window.end,
        chunk_size=config.acquisition.raw.chunk_size,
        overwrite=config.acquisition.overwrite,
        run_pull=lambda scratch_dir: run_raw_pull(
            config,
            output_dir=scratch_dir,
            window=history_window,
            reporter=reporter,
            overwrite=True,
            dry_run=False,
        ),
        reporter=reporter,
    )
    if validation.row_count >= required_history_blocks:
        return result, validation, history_window

    expanded_window = expanded_history_range(
        history_window,
        validation,
        config=config,
        required_history_blocks=required_history_blocks,
    )
    reporter.log(
        "expanding history window backward "
        f"from {history_window.start} to {expanded_window.start} "
        f"for {required_history_blocks} required blocks",
        level="warning",
    )
    expanded_result, expanded_validation = ensure_canonical_raw_dataset(
        chain_name=config.chain.name.value,
        chain_id=config.chain.chain_id,
        output_dir=output_dir,
        expected_start_timestamp=expanded_window.start,
        expected_end_timestamp=expanded_window.end,
        chunk_size=config.acquisition.raw.chunk_size,
        overwrite=True,
        run_pull=lambda scratch_dir: run_raw_pull(
            config,
            output_dir=scratch_dir,
            window=expanded_window,
            reporter=reporter,
            overwrite=True,
            dry_run=False,
        ),
        reporter=reporter,
    )
    if expanded_validation.row_count < required_history_blocks:
        raise ValueError(
            "History dataset is too short after one expansion; "
            f"need at least {required_history_blocks} blocks, "
            f"got {expanded_validation.row_count}"
        )
    return expanded_result, expanded_validation, expanded_window


def ensure_evaluation_raw_dataset(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    evaluation_window: TimestampRange,
    reporter: Reporter,
) -> tuple[CryoRunResult | None, RawPullValidationReport]:
    return ensure_canonical_raw_dataset(
        chain_name=config.chain.name.value,
        chain_id=config.chain.chain_id,
        output_dir=output_dir,
        expected_start_timestamp=evaluation_window.start,
        expected_end_timestamp=evaluation_window.end,
        chunk_size=config.acquisition.raw.chunk_size,
        overwrite=config.acquisition.overwrite,
        run_pull=lambda scratch_dir: run_raw_pull(
            config,
            output_dir=scratch_dir,
            window=evaluation_window,
            reporter=reporter,
            overwrite=True,
            dry_run=False,
        ),
        reporter=reporter,
    )

"""Hydra entrypoint for dataset acquisition and enrichment."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from ..acquisition.cryo import CryoRunResult, evaluation_range, history_range_for_chain, run_cryo
from ..acquisition.enrich import enrich_path
from ..acquisition.raw_normalization import normalize_raw_dataset
from ..acquisition.raw_validation import RawPullValidationReport, validate_raw_pull
from ..acquisition.rpc import Web3BlockClient
from ..core.config import ExperimentConfig, coerce_config
from ..core.console import Reporter, RichReporter
from ..core.tracking import configure_mlflow, log_artifacts, log_config
from ..data.io import iter_block_files, load_enriched_block_frame
from ..data.validation import BlockDatasetValidationReport, validate_exact_window_frame
from ._shared import start_run_if_enabled


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _path_string(path: Path) -> str:
    return path.as_posix()


def _compact_validation_report(
    report: RawPullValidationReport | BlockDatasetValidationReport,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": report.status,
        "rows": report.row_count,
        "block_range": {
            "first": report.first_block_number,
            "last": report.last_block_number,
        },
        "timestamp_range": {
            "first": report.first_timestamp,
            "last": report.last_timestamp,
        },
    }
    if isinstance(report, RawPullValidationReport):
        payload["files"] = report.file_count

    issue_counts: dict[str, int] = {}
    for name in (
        "gap_count",
        "overlap_count",
        "duplicate_count",
        "chain_id_mismatch_count",
        "below_start_count",
        "above_end_count",
    ):
        value = getattr(report, name, 0)
        if value:
            issue_counts[name.removesuffix("_count")] = int(value)
    if issue_counts or report.warnings or report.errors:
        payload["issues"] = {
            **issue_counts,
            **({"warnings": report.warnings} if report.warnings else {}),
            **({"errors": report.errors} if report.errors else {}),
        }
    return payload


def _build_dataset_metadata(
    *,
    config: ExperimentConfig,
    raw_history_dir: Path,
    raw_evaluation_dir: Path,
    enriched_history_dir: Path,
    enriched_evaluation_dir: Path,
    history_window_start: int,
    history_window_end: int,
    evaluation_window_start: int,
    evaluation_window_end: int,
    history_validation: RawPullValidationReport,
    evaluation_validation: RawPullValidationReport,
    history_enriched: BlockDatasetValidationReport,
    evaluation_enriched: BlockDatasetValidationReport,
) -> dict[str, object]:
    return {
        "chain": config.chain.name.value,
        "provider": config.provider.name.value,
        "paths": {
            "raw": {
                "history": _path_string(raw_history_dir),
                "evaluation": _path_string(raw_evaluation_dir),
            },
            "enriched": {
                "history": _path_string(enriched_history_dir),
                "evaluation": _path_string(enriched_evaluation_dir),
            },
        },
        "windows": {
            "history": {
                "start_timestamp": history_window_start,
                "end_timestamp": history_window_end,
            },
            "evaluation": {
                "start_timestamp": evaluation_window_start,
                "end_timestamp": evaluation_window_end,
            },
        },
        "settings": {
            "raw_chunk_size": config.pull.chunk_size,
            "enrich_batch_size": config.pull.enrich_batch_size,
            "max_methods_per_second": config.pull.max_methods_per_second,
        },
        "validation": {
            "raw": {
                "history": _compact_validation_report(history_validation),
                "evaluation": _compact_validation_report(evaluation_validation),
            },
            "enriched": {
                "history": _compact_validation_report(history_enriched),
                "evaluation": _compact_validation_report(evaluation_enriched),
            },
        },
    }


def _has_block_files(path: Path) -> bool:
    try:
        return bool(iter_block_files(path))
    except ValueError:
        return False


def _validate_enriched(
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


def _ensure_canonical_raw_dataset(
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
    if not overwrite and _has_block_files(output_dir):
        validation = validate_raw_pull(
            output_dir,
            expected_chain_name=chain_name,
            expected_chain_id=chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
            expected_chunk_size=chunk_size,
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
        normalize_raw_dataset(
            scratch_dir,
            output_dir,
            chain_name=chain_name,
            expected_chain_id=chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
            chunk_size=chunk_size,
        )

    validation = validate_raw_pull(
        output_dir,
        expected_chain_name=chain_name,
        expected_chain_id=chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
        expected_chunk_size=chunk_size,
    )
    if validation.status != "clean":
        raise ValueError(f"Canonical raw dataset validation failed for {output_dir}: {validation}")
    return pull_result, validation


def _ensure_enriched_dataset(
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
    if not overwrite and _has_block_files(output_dir):
        validation = _validate_enriched(
            output_dir,
            expected_chain_id=expected_chain_id,
            expected_start_timestamp=expected_start_timestamp,
            expected_end_timestamp=expected_end_timestamp,
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
    validation = _validate_enriched(
        output_dir,
        expected_chain_id=expected_chain_id,
        expected_start_timestamp=expected_start_timestamp,
        expected_end_timestamp=expected_end_timestamp,
    )
    if validation.status != "clean":
        raise ValueError(
            f"Canonical enriched dataset validation failed for {output_dir}: {validation}"
        )
    return validation


def run(config: ExperimentConfig, *, reporter: Reporter | None = None) -> None:
    raw_history_dir = Path(config.paths.raw_history_dir)
    raw_evaluation_dir = Path(config.paths.raw_evaluation_dir)
    enriched_history_dir = Path(config.paths.enriched_history_dir)
    enriched_evaluation_dir = Path(config.paths.enriched_evaluation_dir)
    metadata_path = Path(config.paths.dataset_metadata_path)
    if config.tracking.enabled:
        configure_mlflow(config)

    history_window = history_range_for_chain(config.chain)
    evaluation_window = evaluation_range()
    block_client = Web3BlockClient(config.provider, config.chain)
    active_reporter = reporter or RichReporter()
    run_context = start_run_if_enabled(
        config,
        run_name=f"acquire-{config.chain.name.value}-{config.provider.name.value}",
    )
    try:
        if run_context is not None:
            run_context.__enter__()
            log_config(config)
            mlflow.set_tags(config.tracking.tags)

        if config.pull.dry_run:
            with TemporaryDirectory(
                prefix=f"spice-{config.chain.name.value}-history-dry-"
            ) as scratch:
                history_result = run_cryo(
                    config.chain,
                    config.pull,
                    Path(scratch) / "history",
                    history_window,
                    provider=config.provider,
                    overwrite=config.pull.overwrite,
                    dry_run=True,
                    reporter=active_reporter,
                )
            with TemporaryDirectory(
                prefix=f"spice-{config.chain.name.value}-evaluation-dry-"
            ) as scratch:
                evaluation_result = run_cryo(
                    config.chain,
                    config.pull,
                    Path(scratch) / "evaluation",
                    evaluation_window,
                    provider=config.provider,
                    overwrite=config.pull.overwrite,
                    dry_run=True,
                    reporter=active_reporter,
                )
            active_reporter.log(
                json.dumps(
                    {
                        "history_completed_chunks": history_result.completed_chunks,
                        "evaluation_completed_chunks": evaluation_result.completed_chunks,
                        "history_validation": "dry_run",
                        "evaluation_validation": "dry_run",
                    }
                )
            )
            return

        history_result, history_validation = _ensure_canonical_raw_dataset(
            chain_name=config.chain.name.value,
            chain_id=config.chain.chain_id,
            output_dir=raw_history_dir,
            expected_start_timestamp=history_window.start,
            expected_end_timestamp=history_window.end,
            chunk_size=config.pull.chunk_size,
            overwrite=config.pull.overwrite,
            run_pull=lambda scratch_dir: run_cryo(
                config.chain,
                config.pull,
                scratch_dir,
                history_window,
                provider=config.provider,
                overwrite=True,
                dry_run=False,
                reporter=active_reporter,
            ),
            reporter=active_reporter,
        )
        evaluation_result, evaluation_validation = _ensure_canonical_raw_dataset(
            chain_name=config.chain.name.value,
            chain_id=config.chain.chain_id,
            output_dir=raw_evaluation_dir,
            expected_start_timestamp=evaluation_window.start,
            expected_end_timestamp=evaluation_window.end,
            chunk_size=config.pull.chunk_size,
            overwrite=config.pull.overwrite,
            run_pull=lambda scratch_dir: run_cryo(
                config.chain,
                config.pull,
                scratch_dir,
                evaluation_window,
                provider=config.provider,
                overwrite=True,
                dry_run=False,
                reporter=active_reporter,
            ),
            reporter=active_reporter,
        )
        history_enriched = _ensure_enriched_dataset(
            input_dir=raw_history_dir,
            output_dir=enriched_history_dir,
            expected_chain_id=config.chain.chain_id,
            expected_start_timestamp=history_window.start,
            expected_end_timestamp=history_window.end,
            overwrite=config.pull.overwrite,
            fetch_gas_limits=block_client.get_block_gas_limits,
            batch_size=config.pull.enrich_batch_size,
            max_methods_per_second=config.pull.max_methods_per_second,
            reporter=active_reporter,
        )
        evaluation_enriched = _ensure_enriched_dataset(
            input_dir=raw_evaluation_dir,
            output_dir=enriched_evaluation_dir,
            expected_chain_id=config.chain.chain_id,
            expected_start_timestamp=evaluation_window.start,
            expected_end_timestamp=evaluation_window.end,
            overwrite=config.pull.overwrite,
            fetch_gas_limits=block_client.get_block_gas_limits,
            batch_size=config.pull.enrich_batch_size,
            max_methods_per_second=config.pull.max_methods_per_second,
            reporter=active_reporter,
        )
        metadata = _build_dataset_metadata(
            config=config,
            raw_history_dir=raw_history_dir,
            raw_evaluation_dir=raw_evaluation_dir,
            enriched_history_dir=enriched_history_dir,
            enriched_evaluation_dir=enriched_evaluation_dir,
            history_window_start=history_window.start,
            history_window_end=history_window.end,
            evaluation_window_start=evaluation_window.start,
            evaluation_window_end=evaluation_window.end,
            history_validation=history_validation,
            evaluation_validation=evaluation_validation,
            history_enriched=history_enriched,
            evaluation_enriched=evaluation_enriched,
        )
        _write_json(metadata_path, metadata)
        active_reporter.log(
            json.dumps(
                {
                    "history_completed_chunks": (
                        0 if history_result is None else history_result.completed_chunks
                    ),
                    "evaluation_completed_chunks": (
                        0 if evaluation_result is None else evaluation_result.completed_chunks
                    ),
                    "history_validation": history_validation.status,
                    "evaluation_validation": evaluation_validation.status,
                    "history_enriched": history_enriched.status,
                    "evaluation_enriched": evaluation_enriched.status,
                }
            )
        )
        if config.tracking.enabled:
            mlflow.log_metrics(
                {
                    "history_completed_chunks": float(
                        0 if history_result is None else history_result.completed_chunks
                    ),
                    "evaluation_completed_chunks": float(
                        0 if evaluation_result is None else evaluation_result.completed_chunks
                    ),
                    "history_gap_count": float(history_validation.gap_count),
                    "evaluation_gap_count": float(evaluation_validation.gap_count),
                    "history_overlap_count": float(history_validation.overlap_count),
                    "evaluation_overlap_count": float(evaluation_validation.overlap_count),
                }
            )
            log_artifacts(
                [
                    metadata_path,
                ]
            )
    finally:
        if run_context is not None:
            run_context.__exit__(None, None, None)
        if reporter is None:
            active_reporter.close()


@hydra.main(version_base=None, config_path="../conf", config_name="acquire")
def main(cfg: DictConfig) -> None:
    run(coerce_config(cfg, task="acquire"))


if __name__ == "__main__":
    main()

"""Corpus split candidate loading and result envelopes."""

from __future__ import annotations

from pathlib import Path

from ..io import iter_block_files, load_block_frame
from ..metadata import has_block_files
from ..validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
)
from ._policy import CorpusSplitOutcome, SplitDatasetCandidate, SplitDatasetFacts
from ._types import DatasetBuildResult, ExistingDatasetState


def validate_block_dataset(
    path: Path,
    *,
    expected_chain_id: int,
) -> BlockDatasetValidationReport:
    try:
        frame = load_block_frame(path)
    except Exception as exc:  # pragma: no cover - workflow smoke tests cover this path
        return BlockDatasetValidationReport(dataset_path=path, status="error", errors=[str(exc)])
    return validate_contiguous_block_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
    )


def load_existing_dataset(
    path: Path,
    *,
    expected_chain_id: int,
) -> ExistingDatasetState | None:
    if not has_block_files(path):
        return None
    try:
        frame = load_block_frame(path)
    except Exception as exc:
        return ExistingDatasetState(
            path=path,
            validation=BlockDatasetValidationReport(
                dataset_path=path,
                status="error",
                errors=[str(exc)],
            ),
            file_count=len(iter_block_files(path)),
        )
    return ExistingDatasetState(
        path=path,
        validation=validate_contiguous_block_frame(
            frame,
            dataset_path=path,
            expected_chain_id=expected_chain_id,
        ),
        file_count=len(iter_block_files(path)),
    )


def reused_result(
    existing: SplitDatasetCandidate,
    *,
    validation: BlockDatasetValidationReport | None = None,
) -> DatasetBuildResult:
    return DatasetBuildResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=None,
        outcome=CorpusSplitOutcome.REUSED,
    )


def staged_result(
    existing: SplitDatasetCandidate,
    *,
    outcome: CorpusSplitOutcome,
    validation: BlockDatasetValidationReport | None = None,
) -> DatasetBuildResult:
    return DatasetBuildResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=existing.path,
        outcome=outcome,
    )


def split_candidate(existing: ExistingDatasetState | None) -> SplitDatasetCandidate | None:
    if existing is None:
        return None
    return SplitDatasetCandidate(
        path=existing.path,
        validation=existing.validation,
        facts=SplitDatasetFacts(
            status=existing.validation.status,
            first_block_number=existing.validation.first_block_number,
            last_block_number=existing.validation.last_block_number,
        ),
        file_count=existing.file_count,
    )

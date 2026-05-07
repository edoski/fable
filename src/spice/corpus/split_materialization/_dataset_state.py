"""Corpus split candidate loading and result envelopes."""

from __future__ import annotations

from pathlib import Path

from ..io import iter_block_files, load_block_frame
from ..validation import (
    BlockDatasetValidationReport,
    validate_contiguous_block_frame,
)
from ._models import (
    CorpusSplitMaterializationResult,
    CorpusSplitOutcome,
    _SplitDatasetCandidate,
    _SplitDatasetFacts,
)


def validate_block_dataset(
    path: Path,
    *,
    expected_chain_id: int,
    required_columns: frozenset[str] = frozenset(),
) -> BlockDatasetValidationReport:
    try:
        frame = load_block_frame(path)
    except Exception as exc:  # pragma: no cover - workflow smoke tests cover this path
        return BlockDatasetValidationReport(dataset_path=path, status="error", errors=[str(exc)])
    return validate_contiguous_block_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
        required_columns=required_columns,
    )


def load_split_candidate(
    path: Path,
    *,
    expected_chain_id: int,
    required_columns: frozenset[str] = frozenset(),
) -> _SplitDatasetCandidate | None:
    if not _has_block_files(path):
        return None
    try:
        frame = load_block_frame(path)
    except Exception as exc:
        validation = BlockDatasetValidationReport(
            dataset_path=path,
            status="error",
            errors=[str(exc)],
        )
        return _split_candidate(
            path=path,
            validation=validation,
            file_count=len(iter_block_files(path)),
        )
    validation = validate_contiguous_block_frame(
        frame,
        dataset_path=path,
        expected_chain_id=expected_chain_id,
        required_columns=required_columns,
    )
    return _split_candidate(
        path=path,
        validation=validation,
        file_count=len(iter_block_files(path)),
    )


def reused_result(
    existing: _SplitDatasetCandidate,
    *,
    validation: BlockDatasetValidationReport | None = None,
) -> CorpusSplitMaterializationResult:
    return CorpusSplitMaterializationResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=None,
        outcome=CorpusSplitOutcome.REUSED,
    )


def staged_result(
    existing: _SplitDatasetCandidate,
    *,
    outcome: CorpusSplitOutcome,
    validation: BlockDatasetValidationReport | None = None,
) -> CorpusSplitMaterializationResult:
    return CorpusSplitMaterializationResult(
        path=existing.path,
        validation=existing.validation if validation is None else validation,
        file_count=existing.file_count,
        promote_dir=existing.path,
        outcome=outcome,
    )


def _split_candidate(
    *,
    path: Path,
    validation: BlockDatasetValidationReport,
    file_count: int,
) -> _SplitDatasetCandidate:
    return _SplitDatasetCandidate(
        path=path,
        validation=validation,
        facts=_SplitDatasetFacts(
            status=validation.status,
            first_block_number=validation.first_block_number,
            last_block_number=validation.last_block_number,
        ),
        file_count=file_count,
    )


def _has_block_files(path: Path) -> bool:
    try:
        return bool(iter_block_files(path))
    except ValueError:
        return False

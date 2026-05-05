# pyright: strict

"""Workflow-level corpus coverage validation."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import StateConflictError
from ..features import CompiledFeatureContract
from ..temporal.contracts import CompiledProblemContract
from .metadata import DatasetManifest, SplitCoverageMetadata


@dataclass(frozen=True, slots=True)
class CorpusCoverageRequirement:
    history_seconds: int
    history_rows: int
    evaluation_delay_seconds: int | None = None


def training_coverage_requirement(
    contract: CompiledProblemContract,
) -> CorpusCoverageRequirement:
    return CorpusCoverageRequirement(
        history_seconds=contract.required_history_seconds + contract.max_delay_seconds,
        history_rows=contract.warmup_rows + contract.sample_count,
    )


def evaluation_coverage_requirement(
    contract: CompiledProblemContract,
    *,
    delay_seconds: int,
) -> CorpusCoverageRequirement:
    if delay_seconds <= 0:
        raise StateConflictError("Evaluation delay must be positive")
    if delay_seconds > contract.max_delay_seconds:
        raise StateConflictError(
            "Evaluation delay exceeds compiled problem capability: "
            f"{delay_seconds} > {contract.max_delay_seconds}"
        )
    return CorpusCoverageRequirement(
        history_seconds=contract.required_history_seconds,
        history_rows=contract.warmup_rows,
        evaluation_delay_seconds=delay_seconds,
    )


def validate_corpus_coverage(
    manifest: DatasetManifest,
    *,
    contract: CompiledProblemContract,
    feature_contract: CompiledFeatureContract,
    requirement: CorpusCoverageRequirement,
) -> None:
    if feature_contract.feature_prerequisites != contract.feature_prerequisites:
        raise StateConflictError(
            "Feature prerequisites do not match compiled problem prerequisites"
        )
    if manifest.chain.chain_id <= 0:
        raise StateConflictError("Corpus manifest has invalid chain metadata")
    missing_source_columns = (
        set(feature_contract.required_source_columns)
        - manifest.source_requirements.required_columns
    )
    if missing_source_columns:
        raise StateConflictError(
            "Corpus source requirements do not satisfy feature contract: missing "
            + ", ".join(sorted(missing_source_columns))
        )
    _validate_clean_split(manifest, split="history")
    _require_window_span(
        manifest.splits.history.coverage,
        minimum_seconds=requirement.history_seconds,
        label="history coverage",
    )
    if manifest.splits.history.validation.rows < requirement.history_rows:
        raise StateConflictError(
            "Corpus history rows are insufficient for compiled problem: "
            f"{manifest.splits.history.validation.rows} < {requirement.history_rows}"
        )
    if requirement.evaluation_delay_seconds is not None:
        _validate_clean_split(manifest, split="evaluation")
        _require_window_span(
            manifest.splits.evaluation.coverage,
            minimum_seconds=requirement.evaluation_delay_seconds,
            label="evaluation coverage",
        )


def _validate_clean_split(manifest: DatasetManifest, *, split: str) -> None:
    report = (
        manifest.splits.history.validation
        if split == "history"
        else manifest.splits.evaluation.validation
    )
    if report.status != "clean":
        raise StateConflictError(f"Corpus {split} validation is not clean: {report.status}")
    if report.rows <= 0:
        raise StateConflictError(f"Corpus {split} split is empty")


def _require_window_span(
    window: SplitCoverageMetadata,
    *,
    minimum_seconds: int,
    label: str,
) -> None:
    if window.first_timestamp is None or window.last_timestamp is None:
        raise StateConflictError(f"{label} is missing timestamp coverage")
    available_seconds = window.last_timestamp - window.first_timestamp
    if available_seconds < minimum_seconds:
        raise StateConflictError(
            f"{label} is insufficient for compiled workflow: "
            f"{available_seconds}s < {minimum_seconds}s"
        )

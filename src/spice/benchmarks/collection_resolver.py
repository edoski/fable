"""Resolve benchmark evaluate results from remote storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.models import EvaluateConfig, WorkflowTask
from ..core.errors import SpiceOperatorError
from ..execution.transfer import PulledArtifactRoot
from ..modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary
from ..storage.artifact import (
    list_evaluation_summaries,
    load_artifact_manifest,
    load_training_summary,
)
from .models import BenchmarkPlanEntry
from .root_ledger import consumed_artifact_id, consumed_dataset_id
from .runs import BenchmarkSubmissionRecord


@dataclass(frozen=True, slots=True)
class BenchmarkCollectionSelection:
    run_id: str
    storage_root: Path
    artifact_id: str
    evaluation_dataset_id: str
    evaluator_id: str
    configured_delay_seconds: int | None
    execution_ref: str


@dataclass(frozen=True, slots=True)
class ResolvedBenchmarkEvaluation:
    evaluation: LoadedEvaluationSummary
    training: LoadedTrainingSummary | None


def benchmark_collection_selection(
    entry: BenchmarkPlanEntry,
    submission: BenchmarkSubmissionRecord,
) -> BenchmarkCollectionSelection:
    if entry.run_id != submission.run_id:
        raise SpiceOperatorError(
            f"Submission run id does not match benchmark plan entry: "
            f"{submission.run_id} != {entry.run_id}"
        )
    if entry.workflow is not WorkflowTask.EVALUATE:
        raise SpiceOperatorError(f"benchmark run {entry.run_id} is not an evaluate entry")
    if submission.workflow is not WorkflowTask.EVALUATE:
        raise SpiceOperatorError(f"benchmark submission {entry.run_id} is not evaluate")
    if not isinstance(entry.config, EvaluateConfig):
        raise SpiceOperatorError(f"benchmark run {entry.run_id} is not an evaluate config")
    ledger_artifact_id = consumed_artifact_id(entry.root_ledger)
    if ledger_artifact_id is None:
        raise SpiceOperatorError(
            f"benchmark run {entry.run_id} root ledger is missing consumed artifact"
        )
    if ledger_artifact_id != entry.config.artifact_id:
        raise SpiceOperatorError(
            f"benchmark run {entry.run_id} root ledger artifact mismatch: "
            f"{ledger_artifact_id} != {entry.config.artifact_id}"
        )
    ledger_dataset_id = consumed_dataset_id(entry.root_ledger)
    if ledger_dataset_id is None:
        raise SpiceOperatorError(
            f"benchmark run {entry.run_id} root ledger is missing consumed dataset"
        )
    if ledger_dataset_id != entry.config.dataset_id:
        raise SpiceOperatorError(
            f"benchmark run {entry.run_id} root ledger dataset mismatch: "
            f"{ledger_dataset_id} != {entry.config.dataset_id}"
        )
    return BenchmarkCollectionSelection(
        run_id=entry.run_id,
        storage_root=entry.config.storage.root,
        artifact_id=entry.config.artifact_id,
        evaluation_dataset_id=entry.config.dataset_id,
        evaluator_id=entry.config.evaluation.id,
        configured_delay_seconds=entry.config.delay_seconds,
        execution_ref=submission.execution_ref,
    )


def resolve_benchmark_evaluation(
    selection: BenchmarkCollectionSelection,
    *,
    pulled: PulledArtifactRoot,
) -> ResolvedBenchmarkEvaluation | None:
    if pulled.local_record.artifact_id != selection.artifact_id:
        raise SpiceOperatorError(
            "Pulled artifact does not match benchmark collection selection for "
            f"{selection.run_id}: {pulled.local_record.artifact_id} != "
            f"{selection.artifact_id}"
        )
    record = pulled.local_record
    training_summary = load_training_summary(record.state_db_path)
    manifest = load_artifact_manifest(record.state_db_path)
    if manifest.artifact_id != selection.artifact_id:
        raise SpiceOperatorError(
            "Artifact manifest does not match benchmark collection selection for "
            f"{selection.run_id}: {manifest.artifact_id} != {selection.artifact_id}"
        )
    expected_delay = selection.configured_delay_seconds or manifest.max_delay_seconds
    summaries = _matching_evaluation_summaries(
        selection,
        summaries=tuple(list_evaluation_summaries(record.state_db_path)),
        expected_delay=expected_delay,
    )
    if not summaries:
        return None
    return ResolvedBenchmarkEvaluation(
        evaluation=summaries[0],
        training=training_summary,
    )


def _matching_evaluation_summaries(
    selection: BenchmarkCollectionSelection,
    *,
    summaries: tuple[LoadedEvaluationSummary, ...],
    expected_delay: int,
) -> list[LoadedEvaluationSummary]:
    candidates = [
        summary
        for summary in summaries
        if summary.runtime.delay_seconds == expected_delay
        and summary.runtime.evaluator_id == selection.evaluator_id
    ]
    if not candidates:
        return []
    provenance_matches = [
        summary
        for summary in candidates
        if summary.runtime.execution_provenance is not None
        and summary.runtime.execution_provenance.execution_ref == selection.execution_ref
    ]
    if not provenance_matches:
        raise SpiceOperatorError(
            "No evaluation summary matches submitted execution provenance for "
            f"benchmark run {selection.run_id}: expected {selection.execution_ref}"
        )
    if len(provenance_matches) > 1:
        raise SpiceOperatorError(
            "Multiple evaluation summaries match submitted execution provenance for "
            f"benchmark run {selection.run_id}"
        )
    return provenance_matches

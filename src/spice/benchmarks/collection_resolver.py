"""Resolve benchmark evaluate results from remote storage."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import EvaluateConfig
from ..core.errors import SpiceOperatorError
from ..execution.transfer import PulledArtifactRoot
from ..modeling.results import LoadedEvaluationSummary, LoadedTrainingSummary
from ..storage.artifact import (
    list_evaluation_summaries,
    load_artifact_manifest,
    load_training_summary,
)
from .runs import BenchmarkSubmissionRecord


@dataclass(frozen=True, slots=True)
class ResolvedBenchmarkEvaluation:
    evaluation: LoadedEvaluationSummary
    training: LoadedTrainingSummary | None


def resolve_benchmark_evaluation(
    config: EvaluateConfig,
    *,
    pulled: PulledArtifactRoot,
    submission: BenchmarkSubmissionRecord,
) -> ResolvedBenchmarkEvaluation | None:
    record = pulled.local_record
    training_summary = load_training_summary(record.state_db_path)
    manifest = load_artifact_manifest(record.state_db_path)
    expected_delay = config.delay_seconds or manifest.max_delay_seconds
    summaries = [
        summary
        for summary in list_evaluation_summaries(record.state_db_path)
        if summary.runtime.delay_seconds == expected_delay
        and summary.runtime.evaluation_id == config.evaluation.id
    ]
    if not summaries:
        return None
    provenance_matches = [
        summary
        for summary in summaries
        if summary.runtime.execution_provenance is not None
        and summary.runtime.execution_provenance.execution_ref == submission.execution_ref
    ]
    if not provenance_matches:
        raise SpiceOperatorError(
            "No evaluation summary matches submitted execution provenance for "
            f"benchmark run {submission.run_id}: expected {submission.execution_ref}"
        )
    if len(provenance_matches) > 1:
        raise SpiceOperatorError(
            "Multiple evaluation summaries match submitted execution provenance for "
            f"benchmark run {submission.run_id}"
        )
    return ResolvedBenchmarkEvaluation(
        evaluation=provenance_matches[0],
        training=training_summary,
    )

# pyright: strict

"""All-or-nothing benchmark collection from remote completed workflows."""

from __future__ import annotations

from pathlib import Path

from ..config.models import EvaluateConfig, WorkflowTask
from ..core.errors import ConfigResolutionError, SelectorResolutionError, SpiceOperatorError
from ..execution.session import ExecutionSession, open_execution_session
from ..execution.transfer import PulledArtifactRoot, pull_artifact_from_cluster
from .collection_resolver import resolve_benchmark_evaluation
from .result_index import upsert_benchmark_collection_snapshot
from .result_records import (
    BenchmarkCollectionSnapshot,
    BenchmarkResultRecord,
    build_benchmark_result_record,
)
from .result_store import BENCHMARK_RESULT_INDEX_PATH
from .runs import (
    format_datetime,
    load_plan_jsonl,
    load_run_metadata,
    load_submission_jsonl,
    utc_now,
    write_collection_snapshot,
)


def collect_benchmark_run(
    run_dir: Path,
    *,
    index_path: Path = BENCHMARK_RESULT_INDEX_PATH,
) -> BenchmarkCollectionSnapshot:
    metadata = load_run_metadata(run_dir)
    plan = load_plan_jsonl(run_dir)
    submissions = load_submission_jsonl(run_dir)
    evaluate_entries = [entry for entry in plan if entry.workflow is WorkflowTask.EVALUATE]
    collector_time = utc_now()
    session = open_execution_session(metadata.target)
    pulled_artifacts: dict[str, PulledArtifactRoot] = {}
    records: list[BenchmarkResultRecord] = []
    for entry in evaluate_entries:
        if not isinstance(entry.config, EvaluateConfig):
            raise ConfigResolutionError(f"benchmark run {entry.run_id} is not an evaluate config")
        submission = submissions.get(entry.run_id)
        if submission is None:
            raise SpiceOperatorError(f"Missing submission record for benchmark run {entry.run_id}")
        try:
            state = resolve_benchmark_evaluation(
                entry.config,
                pulled=_cached_artifact_pull(
                    entry.config,
                    session=session,
                    cache=pulled_artifacts,
                ),
                submission=submission,
            )
        except SelectorResolutionError as exc:
            raise SpiceOperatorError(str(exc)) from exc
        if state is None:
            raise SpiceOperatorError(
                f"Evaluation summary not found for benchmark run {entry.run_id}"
            )
        records.append(
            build_benchmark_result_record(
                entry=entry,
                submission=submission,
                evaluation=state.evaluation,
                training=state.training,
                collector_time=collector_time,
            )
        )
    snapshot = BenchmarkCollectionSnapshot(
        benchmark=metadata.benchmark,
        run_dir=str(run_dir),
        target=metadata.target,
        run_created_at_utc=metadata.created_at_utc,
        collected_at_utc=format_datetime(collector_time),
        expected_evaluate_count=len(evaluate_entries),
        records=tuple(records),
    )
    write_collection_snapshot(run_dir, snapshot)
    upsert_benchmark_collection_snapshot(snapshot, index_path=index_path)
    return snapshot


def _cached_artifact_pull(
    config: EvaluateConfig,
    *,
    session: ExecutionSession,
    cache: dict[str, PulledArtifactRoot],
) -> PulledArtifactRoot:
    pulled = cache.get(config.artifact_id)
    if pulled is not None:
        return pulled
    pulled = pull_artifact_from_cluster(
        storage_root=config.storage.root,
        session=session,
        artifact_id=config.artifact_id,
        replace=True,
    )
    cache[config.artifact_id] = pulled
    return pulled

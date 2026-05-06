from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from spice.benchmarks.ledger import export_results_csv
from spice.benchmarks.plan_materialization import (
    BenchmarkDependencyLedger,
    BenchmarkRootLedger,
    BenchmarkRootLedgerEntry,
    BenchmarkSelectionLedger,
)
from spice.benchmarks.result_index import (
    list_benchmark_results,
    rebuild_benchmark_result_index,
    upsert_benchmark_collection_snapshot,
)
from spice.benchmarks.result_records import (
    BenchmarkCollectionSnapshot,
    BenchmarkResultRecord,
    MetricValueRecord,
)
from spice.benchmarks.result_store import index_counts
from spice.benchmarks.runs import create_benchmark_run_dir, write_collection_snapshot
from spice.config import WorkflowTask
from spice.core.errors import SpiceOperatorError


def _record(
    *,
    run_id: str,
    metric_id: str = "new_metric",
    recorded_at_utc: str = "2026-05-01T11:00:00Z",
) -> BenchmarkResultRecord:
    return BenchmarkResultRecord(
        run_id=run_id,
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=("case.train",),
            external_slurm_dependencies=(),
            artifact_from_run_id="case.train",
        ),
        dimension_labels={"models": "lstm"},
        selection=BenchmarkSelectionLedger(surface="current_row_fee_dynamics"),
        root_ledger=BenchmarkRootLedger(
            entries=(
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="dataset",
                    root_id="evaluation-dataset-1",
                    dataset_id="evaluation-dataset-1",
                ),
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="artifact",
                    root_id="artifact-1",
                    artifact_id="artifact-1",
                    dataset_id="evaluation-dataset-1",
                    source_run_id="case.train",
                ),
                BenchmarkRootLedgerEntry(
                    run_id=run_id,
                    workflow=WorkflowTask.EVALUATE,
                    role="source",
                    root_kind="dataset",
                    root_id="dataset-1",
                    dataset_id="dataset-1",
                    source_run_id="case.train",
                ),
            ),
        ),
        job_id="42",
        execution_ref="slurm:42",
        git_commit="abc123",
        dependency="afterok:41",
        log_path="/remote/spice-evaluate-42.out",
        evaluation_execution_ref="slurm:42",
        evaluation_job_id="42",
        evaluation_log_path="/remote/spice-evaluate-42.out",
        evaluation_workflow_task="evaluate",
        evaluation_target="disi_l40",
        artifact_id="artifact-1",
        evaluation_storage_id="eval-1",
        artifact_dataset_id="dataset-1",
        artifact_dataset_name="icdcs_2026",
        evaluation_dataset_id="evaluation-dataset-1",
        chain_name="ethereum",
        features_id="core_fee_dynamics",
        model_id="lstm",
        problem_id="current_row_nominal",
        prediction_id="icdcs_2026",
        objective_id="evaluation",
        evaluator_id="poisson_replay",
        delay_seconds=36,
        variant="baseline",
        study_id=None,
        study_name=None,
        recorded_at_utc=recorded_at_utc,
        sample_count=123,
        total_events=7,
        n_history_rows=200,
        n_evaluation_rows=100,
        metrics=(
            MetricValueRecord(source="evaluation", metric_id=metric_id, value=0.12),
            MetricValueRecord(source="training_test", metric_id="total_loss", value=0.3),
        ),
        window_metrics=(),
    )


def _snapshot(
    run_dir: Path,
    *,
    run_id: str,
    recorded_at_utc: str = "2026-05-01T11:00:00Z",
) -> BenchmarkCollectionSnapshot:
    return BenchmarkCollectionSnapshot(
        benchmark="bench",
        run_dir=str(run_dir),
        target="disi_l40",
        run_created_at_utc="2026-05-01T10:00:00Z",
        collected_at_utc="2026-05-01T11:05:00Z",
        expected_evaluate_count=1,
        records=(_record(run_id=run_id, recorded_at_utc=recorded_at_utc),),
    )


def test_result_index_keeps_observations_per_benchmark_run(tmp_path: Path) -> None:
    index_path = tmp_path / "results.sqlite"
    first = _snapshot(tmp_path / "runs" / "bench" / "first", run_id="case.evaluate")
    second = _snapshot(tmp_path / "runs" / "bench" / "second", run_id="case.evaluate")

    upsert_benchmark_collection_snapshot(first, index_path=index_path)
    upsert_benchmark_collection_snapshot(second, index_path=index_path)

    assert index_counts(index_path) == {
        "runs": 2,
        "observations": 2,
        "metrics": 4,
    }
    rows = list_benchmark_results(
        index_path=index_path,
        benchmark="bench",
        chain="ethereum",
        model="lstm",
    )
    assert [row.artifact_id for row in rows] == ["artifact-1", "artifact-1"]
    assert rows[0].chain_name == "ethereum"
    assert rows[0].model_id == "lstm"
    assert rows[0].metrics == {"new_metric": 0.12, "total_loss": 0.3}


def test_result_index_rebuild_is_idempotent_from_run_dirs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = create_benchmark_run_dir("bench", target="disi_l40", runs_root=runs_root)
    write_collection_snapshot(run_dir, _snapshot(run_dir, run_id="case.evaluate"))
    index_path = tmp_path / "results.sqlite"

    first = rebuild_benchmark_result_index(runs_root=runs_root, index_path=index_path)
    second = rebuild_benchmark_result_index(runs_root=runs_root, index_path=index_path)

    assert first == {"runs": 1, "observations": 1, "metrics": 2}
    assert second == first


def test_csv_export_overwrites_from_index(tmp_path: Path) -> None:
    index_path = tmp_path / "results.sqlite"
    output_path = tmp_path / "results.csv"
    output_path.write_text("stale\n", encoding="utf-8")
    upsert_benchmark_collection_snapshot(
        _snapshot(tmp_path / "runs" / "bench" / "one", run_id="case.evaluate"),
        index_path=index_path,
    )

    rows = export_results_csv(output_path=output_path, index_path=index_path)
    exported = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))

    assert len(rows) == 1
    assert exported[0]["artifact_id"] == "artifact-1"
    assert exported[0]["total_loss"] == "0.3"


def test_index_query_and_export_use_normalized_rows_not_payload_json(tmp_path: Path) -> None:
    index_path = tmp_path / "results.sqlite"
    output_path = tmp_path / "results.csv"
    upsert_benchmark_collection_snapshot(
        _snapshot(tmp_path / "runs" / "bench" / "one", run_id="case.evaluate"),
        index_path=index_path,
    )
    with sqlite3.connect(index_path) as connection:
        connection.execute("update result_observations set payload_json = ?", ('{"broken": true}',))

    indexed_rows = list_benchmark_results(
        index_path=index_path,
        benchmark="bench",
    )
    rows = export_results_csv(output_path=output_path, index_path=index_path)

    assert [row.artifact_id for row in indexed_rows] == ["artifact-1"]
    assert rows[0]["artifact_id"] == "artifact-1"
    assert rows[0]["surface"] == "current_row_fee_dynamics"
    with sqlite3.connect(index_path) as connection:
        indexed = connection.execute(
            "select artifact_dataset_id, evaluation_dataset_id from result_observations"
        ).fetchone()
    assert indexed == ("dataset-1", "evaluation-dataset-1")
def test_index_query_rejects_metric_id_collision_across_sources(tmp_path: Path) -> None:
    index_path = tmp_path / "results.sqlite"
    snapshot = _snapshot(tmp_path / "runs" / "bench" / "one", run_id="case.evaluate")
    record = snapshot.records[0].model_copy(
        update={
            "metrics": (
                MetricValueRecord(source="evaluation", metric_id="shared", value=0.12),
                MetricValueRecord(source="training_test", metric_id="shared", value=0.3),
            )
        }
    )
    upsert_benchmark_collection_snapshot(
        snapshot.model_copy(update={"records": (record,)}),
        index_path=index_path,
    )

    with pytest.raises(SpiceOperatorError, match="metric id collision"):
        list_benchmark_results(index_path=index_path, benchmark="bench")


def test_index_list_limits_newest_results(tmp_path: Path) -> None:
    index_path = tmp_path / "results.sqlite"
    for run_id, recorded_at_utc in (
        ("old.evaluate", "2026-05-01T11:00:00Z"),
        ("new.evaluate", "2026-05-01T11:02:00Z"),
        ("middle.evaluate", "2026-05-01T11:01:00Z"),
    ):
        upsert_benchmark_collection_snapshot(
            _snapshot(
                tmp_path / "runs" / "bench" / run_id,
                run_id=run_id,
                recorded_at_utc=recorded_at_utc,
            ),
            index_path=index_path,
        )

    rows = list_benchmark_results(index_path=index_path, benchmark="bench", limit=2)

    assert [row.run_id for row in rows] == ["new.evaluate", "middle.evaluate"]

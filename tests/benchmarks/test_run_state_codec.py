from __future__ import annotations

import json
from pathlib import Path

import pytest

from spice.benchmarks.plan_materialization import (
    BenchmarkDependencyLedger,
    BenchmarkPlanEntry,
    BenchmarkRootFacts,
    BenchmarkRootLedger,
    BenchmarkSelectionLedger,
)
from spice.benchmarks.runs import (
    BenchmarkSubmissionRecord,
    create_benchmark_run,
    load_benchmark_collection_snapshot,
    load_benchmark_run,
    record_benchmark_submission,
)
from spice.config import EvaluateConfig, StorageSpec, WorkflowTask
from spice.config.groups import load_named_group_payload
from spice.core.errors import SpiceOperatorError
from spice.evaluation.registry import coerce_evaluator_config


def _evaluate_config(tmp_path: Path) -> EvaluateConfig:
    return EvaluateConfig(
        storage=StorageSpec(root=tmp_path / "outputs"),
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        evaluation=coerce_evaluator_config(
            load_named_group_payload("poisson_replay", "evaluation")
        ),
        delay_seconds=36,
    )


def _evaluate_entry(tmp_path: Path) -> BenchmarkPlanEntry:
    config = _evaluate_config(tmp_path)
    return BenchmarkPlanEntry(
        run_id="case.evaluate",
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        dependencies=BenchmarkDependencyLedger(
            local_run_ids=("case.train",),
            external_slurm_dependencies=("afterok:42",),
            artifact_from_run_id="case.train",
        ),
        dimension_labels={"features": "core"},
        selection=BenchmarkSelectionLedger(
            evaluation=config.evaluation.id,
            delay_seconds=config.delay_seconds,
        ),
        root_facts=BenchmarkRootFacts(
            consumed_dataset_id=config.dataset_id,
            consumed_artifact_id=config.artifact_id,
            consumed_artifact_dataset_id=config.dataset_id,
        ),
        root_ledger=BenchmarkRootLedger(),
        config=config,
    )


def _run_with_plan(tmp_path: Path):
    return create_benchmark_run(
        "run_state",
        target="disi_l40",
        runs_root=tmp_path / "runs",
        plan=[_evaluate_entry(tmp_path)],
    )


def test_plan_jsonl_keeps_operator_run_state_shape(tmp_path: Path) -> None:
    run = _run_with_plan(tmp_path)
    row = json.loads((run.run_dir / "plan.jsonl").read_text(encoding="utf-8"))

    assert row["dependencies"]["local_run_ids"] == ["case.train"]
    assert row["dimension_labels"] == {"features": "core"}
    assert set(row) >= {"selection", "root_facts", "root_ledger", "config"}
    assert "dataset_id" not in row["selection"]
    assert row["root_facts"]["consumed_dataset_id"] == "dataset-1"
    assert row["config"]["workflow"] == "evaluate"


def test_plan_jsonl_rejects_extra_fields(tmp_path: Path) -> None:
    run = _run_with_plan(tmp_path)
    path = run.run_dir / "plan.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["extra"] = True
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SpiceOperatorError, match="Extra inputs are not permitted"):
        load_benchmark_run(run.run_dir)


def test_plan_jsonl_rejects_non_string_dimension_labels(tmp_path: Path) -> None:
    run = _run_with_plan(tmp_path)
    path = run.run_dir / "plan.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["dimension_labels"]["features"] = 42
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SpiceOperatorError, match="Input should be a valid string"):
        load_benchmark_run(run.run_dir)


def test_submission_jsonl_rejects_extra_fields(tmp_path: Path) -> None:
    run = _run_with_plan(tmp_path)
    record_benchmark_submission(
        run.run_dir,
        BenchmarkSubmissionRecord(
            run_id="case.evaluate",
            workflow=WorkflowTask.EVALUATE,
            job_id="42",
            execution_ref="slurm:42",
            git_commit="abc123",
            dependency=None,
            log_path="/tmp/spice-evaluate-42.out",
        ),
    )
    path = run.run_dir / "submission.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["extra"] = True
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SpiceOperatorError, match="Extra inputs are not permitted"):
        load_benchmark_run(run.run_dir)


def test_collection_snapshot_requires_schema_version(tmp_path: Path) -> None:
    run = create_benchmark_run(
        "collection_schema",
        target="disi_l40",
        runs_root=tmp_path / "runs",
        plan=[],
    )
    (run.run_dir / "collection.json").write_text(
        json.dumps(
            {
                "benchmark": "collection_schema",
                "run_dir": str(run.run_dir),
                "target": "disi_l40",
                "run_created_at_utc": "2026-05-01T10:00:00Z",
                "collected_at_utc": "2026-05-01T11:00:00Z",
                "expected_evaluate_count": 0,
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpiceOperatorError, match="schema_version"):
        load_benchmark_collection_snapshot(run.run_dir)


def test_collection_snapshot_rejects_unknown_schema_version(tmp_path: Path) -> None:
    run = create_benchmark_run(
        "collection_schema",
        target="disi_l40",
        runs_root=tmp_path / "runs",
        plan=[],
    )
    (run.run_dir / "collection.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "benchmark": "collection_schema",
                "run_dir": str(run.run_dir),
                "target": "disi_l40",
                "run_created_at_utc": "2026-05-01T10:00:00Z",
                "collected_at_utc": "2026-05-01T11:00:00Z",
                "expected_evaluate_count": 0,
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpiceOperatorError, match="schema_version"):
        load_benchmark_collection_snapshot(run.run_dir)

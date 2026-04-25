from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from spice.benchmark_runs import (
    BenchmarkSubmissionRecord,
    append_submission_jsonl,
    collect_benchmark_run,
    create_benchmark_run_dir,
    write_plan_jsonl,
)
from spice.config import WorkflowTask
from spice.config.benchmarks import BenchmarkPlanEntry
from spice.core.errors import SelectorResolutionError, SpiceOperatorError


def _write_evaluate_run(run_dir: Path, config) -> None:
    entry = BenchmarkPlanEntry(
        run_id="case.evaluate",
        case_id="case",
        step_id="evaluate",
        workflow=WorkflowTask.EVALUATE,
        depends_on=(),
        external_dependencies=(),
        selection={
            "surface": "same_block_closed",
            "objective": "profit_poisson_replay_2h_mean",
        },
        config=config,
    )
    write_plan_jsonl(run_dir, [entry])
    append_submission_jsonl(
        run_dir,
        BenchmarkSubmissionRecord(
            run_id=entry.run_id,
            workflow=entry.workflow,
            job_id="57549",
            execution_ref="slurm:57549",
            git_commit="abc123",
            dependency=None,
            log_path="/tmp/spice-evaluate-57549.out",
        ),
    )


def _loaded_summary(config):
    return SimpleNamespace(
        evaluation_id="poisson_replay_2h_mean-36s-storage",
        recorded_at=1_700_000_000,
        manifest=SimpleNamespace(
            artifact_id="artifact-1",
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
            feature_set_id=config.feature_set.id,
            model=SimpleNamespace(id=config.model.id),
            problem_id=config.problem.id,
            prediction_id=config.prediction.id,
            objective=config.objective,
            variant=config.artifact.variant,
            study=None,
        ),
        runtime=SimpleNamespace(
            delay_seconds=config.delay_seconds,
            evaluation_id=config.evaluation.id,
            sample_count=123,
            total_events=7,
            metrics=SimpleNamespace(
                values={
                    "profit_over_baseline": 0.12,
                    "cost_over_optimum": 1.5,
                    "baseline_cost_over_optimum": 1.7,
                }
            ),
        ),
    )


def test_benchmark_collect_writes_and_skips_duplicate_rows(
    tmp_path: Path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = load_workflow_config(
        WorkflowTask.EVALUATE,
        workspace=tmp_path,
        surface="same_block_closed",
        variant="baseline",
        delay_seconds=36,
    )
    run_dir = create_benchmark_run_dir(
        "collect_case",
        target="disi_l40",
        git_commit="abc123",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    _write_evaluate_run(run_dir, config)
    artifact_pulls: list[bool] = []

    def fake_pull_artifact(**kwargs):
        artifact_pulls.append(kwargs["replace"])

    monkeypatch.setattr("spice.benchmark_runs.pull_artifact_from_cluster", fake_pull_artifact)
    monkeypatch.setattr(
        "spice.benchmark_runs.load_training_summary",
        lambda path: SimpleNamespace(
            runtime=SimpleNamespace(
                test_metrics=SimpleNamespace(
                    values={
                        "total_loss": 0.3,
                        "offset_accuracy": 0.8,
                    }
                )
            )
        ),
    )
    monkeypatch.setattr(
        "spice.benchmark_runs.list_evaluation_summaries",
        lambda path: [_loaded_summary(config)],
    )
    ledger_path = tmp_path / "benchmarks" / "results.csv"

    records = collect_benchmark_run(
        run_dir=run_dir,
        target_name="disi_l40",
        ledger_path=ledger_path,
        write=True,
    )
    duplicate_records = collect_benchmark_run(
        run_dir=run_dir,
        target_name="disi_l40",
        ledger_path=ledger_path,
        write=True,
    )

    assert records[0].status == "ready"
    assert duplicate_records[0].status == "skipped"
    rows = list(csv.DictReader(ledger_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["git_commit"] == "abc123"
    assert rows[0]["execution_ref"] == "slurm:57549"
    assert rows[0]["surface"] == "same_block_closed"
    assert rows[0]["objective"] == "profit_poisson_replay_2h_mean"
    assert rows[0]["profit_over_baseline"] == "0.12"
    assert rows[0]["total_loss"] == "0.3"
    assert rows[0]["offset_accuracy"] == "0.8"
    assert artifact_pulls == [True, True]


def test_benchmark_collect_refuses_partial_ledger_write(
    tmp_path: Path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = load_workflow_config(
        WorkflowTask.EVALUATE,
        workspace=tmp_path,
        surface="same_block_closed",
        variant="baseline",
        delay_seconds=36,
    )
    run_dir = create_benchmark_run_dir(
        "missing_case",
        target="disi_l40",
        git_commit="abc123",
        runs_root=tmp_path / "outputs" / "benchmarks" / "runs",
    )
    _write_evaluate_run(run_dir, config)
    def missing_remote_artifact(**kwargs):
        del kwargs
        raise SelectorResolutionError(kind="artifact", records=[])

    monkeypatch.setattr("spice.benchmark_runs.pull_artifact_from_cluster", missing_remote_artifact)
    monkeypatch.setattr("spice.benchmark_runs.load_training_summary", lambda path: None)
    monkeypatch.setattr("spice.benchmark_runs.list_evaluation_summaries", lambda path: [])
    ledger_path = tmp_path / "benchmarks" / "results.csv"

    with pytest.raises(SpiceOperatorError, match="Refusing partial"):
        collect_benchmark_run(
            run_dir=run_dir,
            target_name="disi_l40",
            ledger_path=ledger_path,
            write=True,
        )

    assert not ledger_path.exists()

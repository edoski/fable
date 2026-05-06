from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from spice.benchmarks.plan_materialization import (
    BenchmarkDependencyLedger,
    BenchmarkRootLedger,
    BenchmarkRootLedgerEntry,
    BenchmarkSelectionLedger,
)
from spice.benchmarks.result_index import upsert_benchmark_collection_snapshot
from spice.benchmarks.result_records import (
    BenchmarkCollectionSnapshot,
    BenchmarkResultRecord,
    MetricValueRecord,
)
from spice.benchmarks.runs import (
    create_benchmark_run_dir,
    load_benchmark_run,
    write_collection_snapshot,
)
from spice.cli.app import app
from spice.config import WorkflowTask
from spice.core.errors import SpiceOperatorError
from spice.execution.session import ExecutionJobSubmission

runner = CliRunner()


def _write_benchmark(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "benchmark" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _benchmark_record() -> BenchmarkResultRecord:
    return BenchmarkResultRecord(
        run_id="case.evaluate",
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
                    run_id="case.evaluate",
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="dataset",
                    root_id="dataset-1",
                    dataset_id="dataset-1",
                ),
                BenchmarkRootLedgerEntry(
                    run_id="case.evaluate",
                    workflow=WorkflowTask.EVALUATE,
                    role="consumed",
                    root_kind="artifact",
                    root_id="artifact-1",
                    artifact_id="artifact-1",
                    dataset_id="dataset-1",
                    source_run_id="case.train",
                ),
                BenchmarkRootLedgerEntry(
                    run_id="case.evaluate",
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
        evaluation_dataset_id="dataset-1",
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
        recorded_at_utc="2026-05-01T11:00:00Z",
        sample_count=123,
        total_events=7,
        n_history_rows=200,
        n_evaluation_rows=100,
        metrics=(MetricValueRecord(source="evaluation", metric_id="profit", value=0.12),),
        window_metrics=(),
    )


def _write_collection_run(tmp_path: Path) -> Path:
    run_dir = create_benchmark_run_dir(
        "cli_collection",
        target="disi_l40",
        runs_root=tmp_path / "runs",
    )
    write_collection_snapshot(
        run_dir,
        BenchmarkCollectionSnapshot(
            benchmark="cli_collection",
            run_dir=str(run_dir),
            target="disi_l40",
            run_created_at_utc="2026-05-01T10:00:00Z",
            collected_at_utc="2026-05-01T11:00:00Z",
            expected_evaluate_count=1,
            records=(_benchmark_record(),),
        ),
    )
    return run_dir


def test_benchmark_plan_creates_run_dir(isolate_conf_root, tmp_path: Path) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "plan_case",
        {
            "cases": [
                {
                    "id": "single",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "study": "single",
                        "dataset_id": "cor_9a73b1e88edb488afb1e",
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                        }
                    ],
                }
            ]
        },
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "plan_case",
            "--target",
            "disi_l40",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    run_dir = Path(payload["run_dir"])
    assert payload["entries"] == 1
    run = load_benchmark_run(run_dir)
    assert run.metadata.benchmark == "plan_case"
    assert run.plan[0].run_id == "single.train"
    assert run.plan[0].workflow is WorkflowTask.TRAIN
    assert run.plan[0].config.artifact.variant.value == "baseline"


def test_benchmark_submit_uses_persisted_plan(
    isolate_conf_root,
    monkeypatch,
    tmp_path: Path,
) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "submit_case",
        {
            "cases": [
                {
                    "id": "single",
                    "base": {
                        "surface": "current_row_fee_dynamics",
                        "study": "single",
                        "dataset_id": "cor_9a73b1e88edb488afb1e",
                    },
                    "steps": [
                        {
                            "id": "train",
                            "workflow": "train",
                            "set": {"variant": "baseline"},
                            "after": [{"slurm": "afterok:42"}],
                        },
                        {
                            "id": "evaluate",
                            "workflow": "evaluate",
                            "after": ["train"],
                            "artifact_from": "train",
                            "set": {
                                "evaluation": "poisson_replay",
                                "delay_seconds": 12,
                            },
                        },
                    ],
                }
            ]
        },
    )
    calls: list[tuple[str, str, str | None]] = []

    class FakeSession:
        target = SimpleNamespace()

        def remote_git_commit(self) -> str:
            return "abc123"

        def submit_workflow(self, task, *, config, dependency):
            del config
            job_id = str(100 + len(calls))
            calls.append((task.value, "disi_l40", dependency))
            return ExecutionJobSubmission(
                task=task,
                job_id=job_id,
                log_path=Path(f"/tmp/spice-{task.value}-{job_id}.out"),
            )

    monkeypatch.setattr(
        "spice.benchmarks.submission.open_execution_session",
        lambda _target: FakeSession(),
    )
    plan_result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "submit_case",
            "--target",
            "disi_l40",
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert plan_result.exit_code == 0, plan_result.stdout
    run_dir = Path(json.loads(plan_result.stdout)["run_dir"])

    result = runner.invoke(app, ["benchmark", "submit", str(run_dir)])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        ("train", "disi_l40", "afterok:42"),
        ("evaluate", "disi_l40", "afterok:100"),
    ]
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["git_commit"] == "abc123"
    assert rows[0]["execution_ref"] == "slurm:100"
    assert rows[1]["dependency"] == "afterok:100"
    assert Path(rows[0]["run_dir"]) == run_dir
    run = load_benchmark_run(run_dir)
    assert len(run.plan) == 2
    assert sorted(run.submissions) == ["single.evaluate", "single.train"]


def test_benchmark_collect_reports_success(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "spice.benchmarks.collection.collect_benchmark_run",
        lambda _run_dir: SimpleNamespace(records=(object(), object())),
    )

    result = runner.invoke(app, ["benchmark", "collect", str(run_dir)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "run_dir": str(run_dir),
        "records": 2,
        "collection": "complete",
    }


def test_benchmark_collect_failure_writes_no_cli_state(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fail_collect(_run_dir):
        raise SpiceOperatorError("missing evaluation summary")

    monkeypatch.setattr("spice.benchmarks.collection.collect_benchmark_run", fail_collect)

    result = runner.invoke(app, ["benchmark", "collect", str(run_dir)])

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "missing evaluation summary" in result.stderr
    assert not (run_dir / "collection.json").exists()


def test_benchmark_index_export_uses_selected_index(tmp_path: Path) -> None:
    run_dir = _write_collection_run(tmp_path)
    index_path = tmp_path / "custom.sqlite"
    output_path = tmp_path / "results.csv"
    from spice.benchmarks.runs import load_collection_snapshot

    upsert_benchmark_collection_snapshot(
        load_collection_snapshot(run_dir),
        index_path=index_path,
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            "index",
            "export",
            "--index",
            str(index_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {"output": str(output_path), "rows": 1}
    assert index_path.is_file()
    assert "artifact-1" in output_path.read_text(encoding="utf-8")


def test_benchmark_index_commands(tmp_path: Path) -> None:
    run_dir = _write_collection_run(tmp_path)
    index_path = tmp_path / "results.sqlite"

    rebuild = runner.invoke(
        app,
        [
            "benchmark",
            "index",
            "rebuild",
            "--runs-root",
            str(run_dir.parents[1]),
            "--index",
            str(index_path),
        ],
    )
    show = runner.invoke(
        app,
        ["benchmark", "index", "show", "--index", str(index_path)],
    )
    listed = runner.invoke(
        app,
        ["benchmark", "index", "list", "--index", str(index_path), "--limit", "1"],
    )
    exported = runner.invoke(
        app,
        [
            "benchmark",
            "index",
            "export",
            "--index",
            str(index_path),
            "--output",
            str(tmp_path / "export.csv"),
            "--benchmark",
            "cli_collection",
        ],
    )

    assert rebuild.exit_code == 0, rebuild.stdout
    assert json.loads(rebuild.stdout) == {
        "metrics": 1,
        "observations": 1,
        "runs": 1,
    }
    assert show.exit_code == 0, show.stdout
    assert json.loads(show.stdout) == {
        "metrics": 1,
        "observations": 1,
        "runs": 1,
    }
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)["artifact_id"] == "artifact-1"
    assert exported.exit_code == 0, exported.stdout
    assert json.loads(exported.stdout)["rows"] == 1

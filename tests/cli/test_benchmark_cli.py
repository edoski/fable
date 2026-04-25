from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from spice.cli import app
from spice.execution.slurm_ssh import ExecutionJobSubmission

runner = CliRunner()


def _write_benchmark(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "benchmark" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_benchmark_plan_outputs_jsonl(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "plan_case",
        {
            "cases": [
                {
                    "id": "single",
                    "base": {
                        "surface": "same_block_closed",
                        "study": "single",
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

    result = runner.invoke(app, ["benchmark", "plan", "plan_case"])

    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "single.train"
    assert rows[0]["workflow"] == "train"
    assert rows[0]["config"]["artifact"]["variant"] == "baseline"


def test_benchmark_submit_uses_existing_remote_submitter(isolate_conf_root, monkeypatch) -> None:
    conf_root = isolate_conf_root()
    _write_benchmark(
        conf_root,
        "submit_case",
        {
            "cases": [
                {
                    "id": "single",
                    "base": {
                        "surface": "same_block_closed",
                        "study": "single",
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
                            "set": {
                                "objective": "profit_poisson_replay_2h_mean",
                                "evaluation": "poisson_replay_2h_mean",
                                "variant": "baseline",
                                "delay_seconds": 12,
                            },
                        },
                    ],
                }
            ]
        },
    )
    calls: list[tuple[str, str, str | None]] = []

    def fake_submit(task, *, config, target_name, dependency):
        del config
        job_id = str(100 + len(calls))
        calls.append((task.value, target_name, dependency))
        return ExecutionJobSubmission(
            task=task,
            target=SimpleNamespace(),
            job_id=job_id,
            log_path=Path(f"/tmp/spice-{task.value}-{job_id}.out"),
        )

    monkeypatch.setattr("spice.execution.slurm_ssh.submit_execution_workflow", fake_submit)
    monkeypatch.setattr("spice.benchmark_runs.resolve_remote_git_commit", lambda target: "abc123")

    result = runner.invoke(app, ["benchmark", "submit", "submit_case", "--target", "disi_l40"])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        ("train", "disi_l40", "afterok:42"),
        ("evaluate", "disi_l40", "afterok:100"),
    ]
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["git_commit"] == "abc123"
    assert rows[0]["execution_ref"] == "slurm:100"
    assert rows[1]["dependency"] == "afterok:100"
    run_dir = Path(rows[0]["run_dir"])
    assert (run_dir / "plan.jsonl").is_file()
    assert (run_dir / "submission.jsonl").is_file()

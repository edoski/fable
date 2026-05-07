from __future__ import annotations

import stat
from pathlib import Path
from typing import cast

import pytest
import yaml
from typer.testing import CliRunner

from spice.cli.app import app
from spice.config import (
    AcquireConfig,
    EvaluateConfig,
    TrainConfig,
    TrainWorkflowSelection,
    TuneConfig,
    WorkflowTask,
)
from spice.execution.session import ExecutionJobSubmission

runner = CliRunner()


def test_acquire_cli_resolves_selection_surface(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("spice.workflows.acquire.run", _capture)

    result = runner.invoke(
        app,
        [
            "acquire",
            "--surface",
            "current_row_fee_dynamics",
            "--chain",
            "avalanche",
            "--storage-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = cast(AcquireConfig, captured["config"])
    assert config.chain.name == "avalanche"
    assert config.acquisition.dry_run is True
    assert config.storage.root == tmp_path / "outputs"


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (
            "train",
            [
                "--surface",
                "current_row_fee_dynamics",
                "--dataset-id",
                "cor_9a73b1e88edb488afb1e",
                "--study",
                "experiment",
                "--variant",
                "baseline",
            ],
        ),
        (
            "tune",
            [
                "--surface",
                "current_row_fee_dynamics",
                "--dataset-id",
                "cor_9a73b1e88edb488afb1e",
                "--trial-count",
                "2",
            ],
        ),
        (
            "evaluate",
            [
                "--artifact-id",
                "art_test",
                "--dataset-id",
                "cor_9a73b1e88edb488afb1e",
                "--evaluation",
                "poisson_replay",
                "--delay-seconds",
                "12",
            ],
        ),
    ],
)
def test_model_workflow_cli_resolves_and_submits_selection_surface(
    monkeypatch,
    command: str,
    args: list[str],
) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        follow_by_default = False

        def submit_workflow(self, task, *, config, dependency):
            del dependency
            captured["task"] = task
            captured["config"] = config
            return ExecutionJobSubmission(
                task=task,
                job_id="12345",
                log_path=Path("/tmp/spice-job.out"),
            )

    monkeypatch.setattr(
        "spice.execution.submission.open_execution_session",
        lambda _target: FakeSession(),
    )

    result = runner.invoke(
        app,
        [
            command,
            *args,
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = cast(TrainConfig | TuneConfig | EvaluateConfig, captured["config"])
    if isinstance(config, TuneConfig):
        assert config.tuning.trial_count == 2
        return
    if isinstance(config, EvaluateConfig):
        assert config.artifact_id == "art_test"
        assert config.dataset_id == "cor_9a73b1e88edb488afb1e"
        assert config.delay_seconds == 12
        return
    assert config.study.name == "experiment"
    assert config.artifact.variant.value == "baseline"


def test_config_public_commands_only(isolate_conf_root) -> None:
    isolate_conf_root()

    list_result = runner.invoke(app, ["config", "list", "surface"])
    assert list_result.exit_code == 0, list_result.stdout
    assert "current_row_fee_dynamics" in list_result.stdout.splitlines()

    show_result = runner.invoke(app, ["config", "show", "dataset", "icdcs_2026"])
    assert show_result.exit_code == 0, show_result.stdout
    assert yaml.safe_load(show_result.stdout) == {
        "name": "icdcs_2026",
        "evaluation_date": "2025-11-09",
    }

    evaluation_result = runner.invoke(app, ["config", "list", "evaluation"])
    assert evaluation_result.exit_code == 0, evaluation_result.stdout
    assert evaluation_result.stdout.splitlines() == [
        "full_temporal_replay",
        "poisson_replay",
    ]

    builder_result = runner.invoke(app, ["config", "list", "dataset-builder"])
    assert builder_result.exit_code == 0, builder_result.stdout
    assert builder_result.stdout.splitlines() == ["fixed_sequence_temporal"]

    prediction_result = runner.invoke(app, ["config", "list", "prediction"])
    assert prediction_result.exit_code == 0, prediction_result.stdout
    assert prediction_result.stdout.splitlines() == ["icdcs_2026"]


def test_config_edit_seeds_missing_file_and_uses_editor(
    tmp_path: Path,
    isolate_conf_root,
    monkeypatch,
) -> None:
    conf_root = isolate_conf_root()
    log_path = tmp_path / "editor.log"
    editor_path = tmp_path / "fake-editor"
    editor_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f'echo "$1" > "{log_path}"',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    editor_path.chmod(editor_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("EDITOR", str(editor_path))
    monkeypatch.delenv("VISUAL", raising=False)

    result = runner.invoke(app, ["config", "edit", "problem", "phase2_problem"])

    assert result.exit_code == 0, result.stdout
    created_path = conf_root / "problem" / "phase2_problem.yaml"
    assert created_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() == str(created_path)
    assert yaml.safe_load(created_path.read_text(encoding="utf-8"))["id"] == "phase2_problem"


def test_train_submit_uses_cli_default_remote_target(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        follow_by_default = False

        def submit_workflow(self, task, *, config, dependency):
            del config, dependency
            captured["task"] = task
            return ExecutionJobSubmission(
                task=task,
                job_id="12345",
                log_path=Path("/tmp/spice-train-12345.out"),
            )

    def fake_open_session(target_name: str) -> FakeSession:
        captured["target_name"] = target_name
        return FakeSession()

    monkeypatch.setattr("spice.execution.submission.open_execution_session", fake_open_session)

    result = runner.invoke(
        app,
        [
            "train",
            "--surface",
            "current_row_fee_dynamics",
            "--dataset-id",
            "cor_9a73b1e88edb488afb1e",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"task": WorkflowTask.TRAIN, "target_name": "disi_l40"}


def test_acquire_rejects_objective_option() -> None:
    result = runner.invoke(
        app,
        [
            "acquire",
            "--surface",
            "current_row_fee_dynamics",
            "--objective",
            "validation_total_loss",
        ],
    )

    assert result.exit_code != 0
    assert "--objective" in result.output


def test_train_submit_cli_renders_follow_failure(monkeypatch) -> None:
    from spice.cli.commands import workflows as workflow_commands

    def _fake_resolve(selection) -> object:
        assert isinstance(selection, TrainWorkflowSelection)
        assert selection.surface == "current_row_fee_dynamics"
        assert selection.dataset_id == "cor_9a73b1e88edb488afb1e"
        return TrainConfig.model_construct(workflow=WorkflowTask.TRAIN)

    class FakeSession:
        follow_by_default = True

        def submit_workflow(
            self,
            task: WorkflowTask,
            *,
            config,
            dependency: str | None = None,
        ) -> ExecutionJobSubmission:
            del config, dependency
            return ExecutionJobSubmission(
                task=task,
                job_id="12345",
                log_path=Path("/remote/logs/spice-train-12345.out"),
            )

        def follow_job(self, _submission: ExecutionJobSubmission) -> str:
            return "FAILED"

    monkeypatch.setattr(workflow_commands, "resolve_workflow_config", _fake_resolve)
    monkeypatch.setattr(
        "spice.execution.submission.open_execution_session",
        lambda _target: FakeSession(),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--surface",
            "current_row_fee_dynamics",
            "--dataset-id",
            "cor_9a73b1e88edb488afb1e",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "submit workflow=train job_id=12345" in result.stdout
    assert "submit finished job_id=12345 state=FAILED" in result.stdout
    assert "Job 12345 ended with state FAILED" in result.stderr

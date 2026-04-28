from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from typer.testing import CliRunner

from spice.cli import app
from spice.config import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.execution.slurm_ssh import ExecutionJobSubmission
from spice.storage.layout import resolve_workflow_paths

runner = CliRunner()


def test_acquire_cli_resolves_request_surface(tmp_path, monkeypatch) -> None:
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
    paths = resolve_workflow_paths(config)
    assert config.chain.name == "avalanche"
    assert config.acquisition.dry_run is True
    assert paths.output_root == tmp_path / "outputs"


@pytest.mark.parametrize(
    ("command", "runner_path", "args"),
    [
        (
            "train",
            "spice.workflows.train.run",
            ["--study", "experiment", "--variant", "baseline"],
        ),
        (
            "tune",
            "spice.workflows.tune.run",
            ["--trial-count", "2"],
        ),
        (
            "evaluate",
            "spice.workflows.evaluate.run",
            ["--study", "experiment", "--variant", "baseline", "--delay-seconds", "12"],
        ),
    ],
)
def test_model_workflow_cli_resolves_local_request_surface(
    tmp_path: Path,
    monkeypatch,
    command: str,
    runner_path: str,
    args: list[str],
) -> None:
    captured: dict[str, object] = {}

    def _capture(config) -> None:
        captured["config"] = config

    monkeypatch.setattr(runner_path, _capture)

    result = runner.invoke(
        app,
        [
            command,
            "--surface",
            "current_row_fee_dynamics",
            *args,
            "--storage-root",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    config = cast(TrainConfig | TuneConfig | EvaluateConfig, captured["config"])
    assert resolve_workflow_paths(config).output_root == tmp_path / "outputs"
    if isinstance(config, TuneConfig):
        assert config.tuning.trial_count == 2
        return
    assert config.study.name == "experiment"
    assert config.artifact.variant.value == "baseline"
    if isinstance(config, EvaluateConfig):
        assert config.delay_seconds == 12


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
    assert "poisson_replay_2h_mean" in evaluation_result.stdout.splitlines()
    assert "poisson_replay_2h_total" in evaluation_result.stdout.splitlines()

    acquisition_result = runner.invoke(app, ["config", "list", "acquisition"])
    assert acquisition_result.exit_code != 0

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
                f'echo \"$1\" > \"{log_path}\"',
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


def test_train_submit_rejects_local_storage_override(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--submit", "--storage-root", str(tmp_path)])

    assert result.exit_code != 0
    assert "--storage-root cannot be combined with --submit" in result.output


def test_train_submit_uses_cli_default_remote_target(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_submit(task, *, config, target_name, dependency):
        del config, dependency
        captured["task"] = task
        captured["target_name"] = target_name
        return ExecutionJobSubmission(
            task=task,
            target=SimpleNamespace(spec=SimpleNamespace(follow_by_default=False)),
            job_id="12345",
            log_path=Path("/tmp/spice-train-12345.out"),
        )

    monkeypatch.setattr(
        "spice.cli.commands.workflows.submit_execution_workflow",
        fake_submit,
    )

    result = runner.invoke(app, ["train", "--surface", "current_row_fee_dynamics", "--submit"])

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


def test_surface_option_replaces_preset_option() -> None:
    surface_result = runner.invoke(
        app,
        ["train", "--surface", "current_row_fee_dynamics", "--help"],
    )
    preset_result = runner.invoke(app, ["train", "--preset", "current_row_fee_dynamics"])

    assert surface_result.exit_code == 0, surface_result.output
    assert preset_result.exit_code != 0
    assert "--preset" in preset_result.output


def test_train_submit_cli_preflights_and_routes_to_execution_backend(
    monkeypatch,
) -> None:
    from spice.cli.commands import workflows as workflow_commands

    events: list[tuple[str, object]] = []

    def _fake_resolve(task: WorkflowTask, request) -> object:
        events.append(("resolve", (task, request)))
        return SimpleNamespace(
            study=SimpleNamespace(name=request.study),
            artifact=SimpleNamespace(variant=SimpleNamespace(value=request.variant)),
        )

    def _fake_submit(
        task: WorkflowTask,
        *,
        config,
        target_name: str = "disi_l40",
        dependency: str | None = None,
    ) -> ExecutionJobSubmission:
        events.append(("submit", (task, config, target_name, dependency)))
        return ExecutionJobSubmission(
            task=task,
            target=SimpleNamespace(spec=SimpleNamespace(follow_by_default=False)),
            job_id="12345",
            log_path=Path("/remote/logs/spice-train-12345.out"),
        )

    monkeypatch.setattr(workflow_commands, "resolve_workflow_config", _fake_resolve)
    monkeypatch.setattr(workflow_commands, "submit_execution_workflow", _fake_submit)

    result = runner.invoke(
        app,
        [
            "train",
            "--surface",
            "current_row_fee_dynamics",
            "--submit",
            "--study",
            "default",
            "--variant",
            "baseline",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert [event[0] for event in events] == ["resolve", "submit"]
    resolved_task, request = cast(tuple[WorkflowTask, object], events[0][1])
    assert resolved_task is WorkflowTask.TRAIN
    assert request.study == "default"
    assert request.variant == "baseline"
    submitted_task, submitted_config, target_name, dependency = cast(
        tuple[WorkflowTask, object, str, str | None],
        events[1][1],
    )
    assert submitted_task is WorkflowTask.TRAIN
    assert target_name == "disi_l40"
    assert dependency is None
    assert (
        "submit workflow=train job_id=12345 log=/remote/logs/spice-train-12345.out"
        in result.stdout
    )

from __future__ import annotations

from typer.testing import CliRunner

from spice.cli import app

runner = CliRunner()


def test_root_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout
    assert "SPICE workflow CLI." in result.stdout
    assert "config" in result.stdout
    assert "show" in result.stdout
    assert "delete" in result.stdout
    assert "acquire" in result.stdout
    assert "train" in result.stdout
    assert "tune" in result.stdout
    assert "evaluate" in result.stdout


def test_acquire_help_includes_panels_and_example() -> None:
    result = runner.invoke(app, ["acquire", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Selection" in result.stdout
    assert "Outputs" in result.stdout
    assert "Execution" in result.stdout
    assert "Example:" in result.stdout
    assert "--preset" in result.stdout
    assert "--problem" in result.stdout
    assert "--task" not in result.stdout
    assert "--feature-set" in result.stdout
    assert "--provider" in result.stdout


def test_main_workflow_help_stays_operator_focused() -> None:
    for command in ("train", "tune", "evaluate", "show"):
        result = runner.invoke(app, [command, "--help"])

        assert result.exit_code == 0, result.stdout
        assert "Example:" in result.stdout


def test_config_help_lists_core_authoring_commands() -> None:
    result = runner.invoke(app, ["config", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "list" in result.stdout
    assert "show" in result.stdout
    assert "edit" in result.stdout
    assert "create" not in result.stdout
    assert "update" not in result.stdout
    assert "delete" not in result.stdout

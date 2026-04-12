from __future__ import annotations

from typer.testing import CliRunner

from spice.cli import app

runner = CliRunner()


def test_root_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout
    assert "SPICE workflow CLI." in result.stdout
    assert "acquire" in result.stdout
    assert "train" in result.stdout
    assert "tune" in result.stdout
    assert "simulate" in result.stdout


def test_acquire_help_includes_panels_and_example() -> None:
    result = runner.invoke(app, ["acquire", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Selection" in result.stdout
    assert "Overrides" in result.stdout
    assert "Profiles" in result.stdout
    assert "Outputs" in result.stdout
    assert "Execution" in result.stdout
    assert "Example:" in result.stdout
    assert "--preset" in result.stdout
    assert "--provider" in result.stdout

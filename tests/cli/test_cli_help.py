from __future__ import annotations

import pytest
from typer.testing import CliRunner

from spice.cli import app

runner = CliRunner()

_REMOVED_WORKFLOW_FLAGS = {
    "--dataset",
    "--provider",
    "--model",
    "--dataset-builder",
    "--prediction",
}


def test_root_help_lists_public_command_surface() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout
    for token in (
        "config",
        "reconstruct",
        "show",
        "delete",
        "acquire",
        "train",
        "tune",
        "evaluate",
    ):
        assert token in result.stdout


@pytest.mark.parametrize(
    ("command", "expected_flags"),
    [
        (
            "acquire",
            {"--preset", "--chain", "--problem", "--feature-set", "--storage-root", "--dry-run"},
        ),
        (
            "train",
            {
                "--preset",
                "--chain",
                "--problem",
                "--feature-set",
                "--study",
                "--variant",
                "--submit",
            },
        ),
        (
            "tune",
            {"--preset", "--chain", "--problem", "--feature-set", "--trial-count", "--submit"},
        ),
        (
            "evaluate",
            {
                "--preset",
                "--chain",
                "--problem",
                "--feature-set",
                "--evaluation",
                "--study",
                "--variant",
                "--delay-seconds",
                "--submit",
            },
        ),
    ],
)
def test_workflow_help_exposes_only_the_reduced_surface(
    command: str,
    expected_flags: set[str],
) -> None:
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0, result.stdout
    for flag in expected_flags:
        assert flag in result.stdout
    for flag in _REMOVED_WORKFLOW_FLAGS:
        assert flag not in result.stdout


def test_config_list_help_shows_public_groups_only() -> None:
    result = runner.invoke(app, ["config", "list", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "preset, dataset, chain, problem, provider" in result.stdout
    assert "evaluation" not in result.stdout
    assert "model" not in result.stdout
    assert "tuning-space" not in result.stdout


def test_train_submit_rejects_local_storage_override(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["train", "--submit", "--storage-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "--storage-root cannot be combined with --submit" in result.output

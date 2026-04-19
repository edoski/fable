from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from typer.testing import CliRunner

from spice.cli import app
from spice.remote.workflows import RemoteJobSubmission
from spice.storage.catalog import CatalogArtifactRecord

runner = CliRunner()


def test_config_list_remote_routes_to_remote_cli(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("spice.cli.commands.remote.resolve_remote_target", lambda: object())

    def fake_run_remote_cli(_target, args: list[str]):
        captured["args"] = args
        return CompletedProcess(args=args, returncode=0, stdout="execution\n", stderr="")

    monkeypatch.setattr("spice.cli.commands.remote.run_remote_cli", fake_run_remote_cli)

    result = runner.invoke(app, ["remote", "config", "list", "execution"])

    assert result.exit_code == 0, result.stdout
    assert captured["args"] == ["config", "list", "execution"]
    assert result.stdout == "execution\n"


def test_show_artifact_remote_routes_to_remote_cli(monkeypatch) -> None:
    captured: dict[str, object] = {}
    target = SimpleNamespace(
        spec=SimpleNamespace(paths=SimpleNamespace(storage_root=Path("/remote-storage")))
    )

    monkeypatch.setattr("spice.cli.commands.remote.resolve_remote_target", lambda: target)

    def fake_run_remote_cli(_target, args: list[str]):
        captured["args"] = args
        return CompletedProcess(args=args, returncode=0, stdout="artifact summary\n", stderr="")

    monkeypatch.setattr("spice.cli.commands.remote.run_remote_cli", fake_run_remote_cli)

    result = runner.invoke(
        app,
        [
            "remote",
            "show",
            "artifact",
            "--chain",
            "ethereum",
            "--dataset",
            "icdcs_2026",
            "--feature-set",
            "icdcs_2026",
            "--prediction",
            "candidate_offset_selection",
            "--model",
            "lstm",
            "--problem",
            "icdcs_2026",
            "--variant",
            "baseline",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["args"] == [
        "show",
        "artifact",
        "--chain",
        "ethereum",
        "--dataset",
        "icdcs_2026",
        "--feature-set",
        "icdcs_2026",
        "--prediction",
        "candidate_offset_selection",
        "--model",
        "lstm",
        "--problem",
        "icdcs_2026",
        "--variant",
        "baseline",
        "--storage-root",
        "/remote-storage",
    ]


def test_train_remote_detach_submits_without_follow(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_submit_remote_workflow(
        task,
        *,
        cli_args: list[str],
        execution_name: str | None = None,
    ):
        captured["task"] = task
        captured["cli_args"] = cli_args
        captured["execution_name"] = execution_name
        return RemoteJobSubmission(
            task=task,
            execution_name=execution_name or "disi_l40",
            target=SimpleNamespace(spec=SimpleNamespace(follow_by_default=True)),
            job_id="12345",
            log_path=Path("/tmp/spice-train-12345.out"),
        )

    monkeypatch.setattr(
        "spice.cli.commands.remote.submit_remote_workflow",
        fake_submit_remote_workflow,
    )

    result = runner.invoke(
        app,
        [
            "remote",
            "train",
            "--preset",
            "icdcs_2026",
            "--prediction",
            "candidate_offset_selection",
            "--detach",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["execution_name"] == "disi_l40"
    assert captured["cli_args"] == [
        "--preset",
        "icdcs_2026",
        "--prediction",
        "candidate_offset_selection",
    ]
    assert "submitted remote train" in result.stdout


def test_refresh_catalog_remote_routes_to_remote_cli(monkeypatch) -> None:
    captured: dict[str, object] = {}
    target = SimpleNamespace(
        spec=SimpleNamespace(paths=SimpleNamespace(storage_root=Path("/remote-storage")))
    )

    monkeypatch.setattr("spice.cli.commands.remote.resolve_remote_target", lambda: target)

    def fake_run_remote_cli(_target, args: list[str]):
        captured["args"] = args
        return CompletedProcess(args=args, returncode=0, stdout="catalog refreshed\n", stderr="")

    monkeypatch.setattr("spice.cli.commands.remote.run_remote_cli", fake_run_remote_cli)

    result = runner.invoke(app, ["remote", "refresh", "catalog"])

    assert result.exit_code == 0, result.stdout
    assert captured["args"] == [
        "refresh",
        "catalog",
        "--storage-root",
        "/remote-storage",
    ]
    assert result.stdout == "catalog refreshed\n"


def test_pull_artifact_warns_when_matching_dataset_is_missing(monkeypatch, tmp_path: Path) -> None:
    record = CatalogArtifactRecord(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="icdcs_2026",
        chain_name="ethereum",
        feature_set_id="icdcs_2026",
        prediction_id="candidate_offset_selection",
        model_id="lstm",
        problem_id="icdcs_2026",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=tmp_path / "remote-artifact",
        state_db_path=tmp_path / "remote-artifact" / ".spice" / "state.sqlite",
    )

    monkeypatch.setattr(
        "spice.cli.commands.transfer.pull_artifact_from_remote",
        lambda **_kwargs: (record, False),
    )

    result = runner.invoke(
        app,
        [
            "pull",
            "artifact",
            "--chain",
            "ethereum",
            "--dataset",
            "icdcs_2026",
            "--feature-set",
            "icdcs_2026",
            "--prediction",
            "candidate_offset_selection",
            "--model",
            "lstm",
            "--problem",
            "icdcs_2026",
            "--variant",
            "baseline",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "pulled artifact artifact-1 into local storage" in result.stdout
    assert "warning: matching local dataset root is missing" in result.stderr

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from typer.testing import CliRunner

from spice.cli import app
from spice.execution.transfer import PulledArtifactRoot
from spice.storage.catalog import CatalogArtifactRecord, CatalogDatasetRecord

runner = CliRunner()


def _dataset_record(root_path: Path) -> CatalogDatasetRecord:
    return CatalogDatasetRecord(
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _artifact_record(root_path: Path) -> CatalogArtifactRecord:
    return CatalogArtifactRecord(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def test_transfer_push_dataset_command_routes_to_dataset_transfer(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    record = _dataset_record(tmp_path / "outputs" / "corpora" / "ethereum" / "dataset-1")
    monkeypatch.setattr(
        "spice.cli.commands.transfer.open_execution_session",
        lambda target: captured.setdefault("target", target) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spice.cli.commands.transfer.push_dataset_to_cluster",
        lambda **kwargs: captured.update({"kwargs": kwargs}) or record,
    )

    result = runner.invoke(
        app,
        [
            "transfer",
            "push",
            "dataset",
            "--dataset-id",
            record.dataset_id,
            "--storage-root",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["target"] == "disi_l40"
    assert cast(dict[str, object], captured["kwargs"])["dataset_id"] == record.dataset_id
    assert "push dataset=dataset dataset_id=dataset-1" in result.stdout


def test_transfer_pull_artifact_command_uses_pulled_envelope(monkeypatch, tmp_path) -> None:
    record = _artifact_record(tmp_path / "outputs" / "artifacts" / "ethereum" / "artifact-1")
    pulled = PulledArtifactRoot(
        source_record=record,
        local_record=record,
        destination_root=record.root_path,
        dataset_present=False,
    )
    monkeypatch.setattr(
        "spice.cli.commands.transfer.open_execution_session",
        lambda _target: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spice.cli.commands.transfer.pull_artifact_from_cluster",
        lambda **_kwargs: pulled,
    )

    result = runner.invoke(app, ["transfer", "pull", "artifact", "--artifact-id", "artifact-1"])

    assert result.exit_code == 0, result.stdout
    assert "pull artifact=artifact-1" in result.stdout
    assert "matching local dataset root is missing" in result.stderr


def test_transfer_study_commands_are_removed() -> None:
    push_result = runner.invoke(app, ["transfer", "push", "study", "--study-id", "study-1"])
    pull_result = runner.invoke(app, ["transfer", "pull", "study", "--study-id", "study-1"])

    assert push_result.exit_code != 0
    assert pull_result.exit_code != 0

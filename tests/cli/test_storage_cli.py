from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spice.cli import app
from spice.cli.commands import storage as storage_commands
from spice.storage.catalog.store import upsert_artifact_record
from spice.storage.layout import catalog_db_path

runner = CliRunner()


def _write_artifact_record(
    storage_root: Path,
    artifact_id: str,
    *,
    model_id: str = "model",
) -> None:
    root_path = storage_root / "artifacts" / "ethereum" / artifact_id
    upsert_artifact_record(
        catalog_db_path(storage_root),
        artifact_id=artifact_id,
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id=model_id,
        problem_id="problem",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def test_show_writes_success_to_stdout_and_ambiguous_detail_to_stderr(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    _write_artifact_record(storage_root, "art_1")
    _write_artifact_record(storage_root, "art_2")

    success = runner.invoke(app, ["show", "artifact", "--storage-root", str(storage_root)])

    assert success.exit_code == 0, success.output
    assert "artifact list" in success.stdout
    assert "art_1" in success.stdout

    failure = runner.invoke(
        app,
        ["show", "artifact", "--storage-root", str(storage_root), "--detail", "epochs"],
    )

    assert failure.exit_code != 0
    assert "artifact matches" not in failure.stdout
    assert "--detail requires exactly one artifact match" in failure.stderr


def test_show_detail_uses_unique_filtered_match(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "outputs"
    _write_artifact_record(storage_root, "art_1", model_id="lstm")
    _write_artifact_record(storage_root, "art_2", model_id="transformer")
    seen: dict[str, object] = {}

    def fake_show_root_detail(root_path: Path, *, detail: str | None) -> None:
        seen.update({"root_path": root_path, "detail": detail})

    monkeypatch.setattr(storage_commands, "_show_root_detail", fake_show_root_detail)

    result = runner.invoke(
        app,
        [
            "show",
            "artifact",
            "--storage-root",
            str(storage_root),
            "--model",
            "lstm",
            "--detail",
            "epochs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen == {
        "root_path": storage_root / "artifacts" / "ethereum" / "art_1",
        "detail": "epochs",
    }

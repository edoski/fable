from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spice.cli.app import app
from spice.cli.commands import storage as storage_commands
from spice.storage.catalog.index import upsert_catalog_record
from spice.storage.operator import RenderableSections, StorageShowRendered
from tests.catalog_helpers import artifact_record

runner = CliRunner()


def _write_artifact_record(
    storage_root: Path,
    artifact_id: str,
    *,
    model_id: str = "model",
) -> None:
    root_path = storage_root / "artifacts" / "ethereum" / artifact_id
    upsert_catalog_record(
        storage_root,
        artifact_record(root_path, artifact_id=artifact_id, dataset_id="cor_1", model_id=model_id),
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
    assert isinstance(failure.exception, SystemExit)
    assert "artifact matches" not in failure.stdout
    assert "--detail requires exactly one artifact match" in failure.stderr


def test_show_detail_uses_unique_filtered_match(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "outputs"
    seen: dict[str, object] = {}

    def fake_show_storage(query):
        seen.update(
            {
                "storage_root": query.storage_root,
                "kind": query.kind,
                "model": query.selector.model_id,
                "detail": query.detail,
            }
        )
        return StorageShowRendered(
            RenderableSections(
                title="artifact summary",
                sections=[("artifact", [("artifact id", "art_1")])],
            )
        )

    monkeypatch.setattr(storage_commands, "show_storage", fake_show_storage)

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
        "storage_root": storage_root,
        "kind": "artifact",
        "model": "lstm",
        "detail": "epochs",
    }

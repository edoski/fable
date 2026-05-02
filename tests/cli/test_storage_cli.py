from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spice.cli import app
from spice.cli.commands import storage as storage_commands
from spice.storage.catalog import CatalogArtifactRecord
from spice.storage.catalog.registry import ARTIFACT_ROOT_SPEC
from spice.storage.layout import catalog_db_path
from spice.storage.operator import RenderableSections, StorageShowRendered

runner = CliRunner()


def _write_artifact_record(
    storage_root: Path,
    artifact_id: str,
    *,
    model_id: str = "model",
) -> None:
    root_path = storage_root / "artifacts" / "ethereum" / artifact_id
    ARTIFACT_ROOT_SPEC.upsert(
        catalog_db_path(storage_root),
        CatalogArtifactRecord(
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
        ),
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
    seen: dict[str, object] = {}

    def fake_show_storage(query):
        seen.update(
            {
                "storage_root": query.storage_root,
                "kind": query.kind,
                "model": query.selector.model_id,
                "detail": query.detail,
                "has_filters": query.has_filters,
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
        "has_filters": True,
    }

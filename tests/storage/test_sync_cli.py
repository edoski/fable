from __future__ import annotations

from pathlib import Path

from spice.storage import sync_cli
from spice.storage.catalog.codecs import decode_remote_catalog_record
from spice.storage.engine import RootKind
from tests.catalog_helpers import dataset_record


def test_sync_cli_resolve_record_emits_remote_catalog_envelope(monkeypatch, capsys) -> None:
    root = Path("/storage/corpora/ethereum/dataset-1")
    record = dataset_record(root)
    monkeypatch.setattr(
        sync_cli,
        "resolve_catalog_record_by_id",
        lambda _storage_root, *, root_kind, root_id: record,
    )

    sync_cli.main(
        [
            "resolve-record",
            "--storage-root",
            "/storage",
            "--root-kind",
            "corpus",
            "--root-id",
            "dataset-1",
        ]
    )

    decoded = decode_remote_catalog_record(capsys.readouterr().out)
    assert decoded == record


def test_sync_cli_finalize_stage_uses_root_kind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_promote_root_stage(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sync_cli, "promote_root_stage", fake_promote_root_stage)

    sync_cli.main(
        [
            "finalize-stage",
            "--storage-root",
            "/storage",
            "--destination-root",
            "/storage/corpora/ethereum/dataset-1",
            "--staged-root",
            "/storage/corpora/ethereum/.dataset-1.incoming",
            "--root-kind",
            "corpus",
            "--replace",
        ]
    )

    assert captured["expected_root_kind"] is RootKind.CORPUS
    assert captured["replace"] is True

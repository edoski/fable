from __future__ import annotations

import pytest

from spice.core.errors import StateLayoutError
from spice.storage.catalog.index import list_artifact_records, upsert_catalog_record
from spice.storage.engine import RootKind, ensure_state_db, state_db_path
from spice.storage.lifecycle import delete_catalog_artifact_root
from tests.catalog_helpers import artifact_record


def _artifact_record(root_path):
    return artifact_record(
        root_path,
        artifact_id="art_1",
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        variant="baseline",
        study_id=None,
        study_name=None,
    )


@pytest.mark.parametrize("case", ["outside", "kind_mismatch", "missing_db"])
def test_delete_artifact_rejects_invalid_catalog_roots(tmp_path, case: str) -> None:
    storage_root = tmp_path / "outputs"
    if case == "outside":
        artifact_root = tmp_path / "outside-artifact"
        artifact_root.mkdir()
        message = "outside the artifact storage subtree"
    else:
        artifact_root = storage_root / "artifacts" / "ethereum" / "art_1"
        if case == "kind_mismatch":
            ensure_state_db(state_db_path(artifact_root), root_kind=RootKind.STUDY, tables=())
            message = "Catalog root kind mismatch"
        else:
            artifact_root.mkdir(parents=True)
            message = "missing state DB"

    with pytest.raises(StateLayoutError, match=message):
        delete_catalog_artifact_root(storage_root, _artifact_record(artifact_root))

    assert artifact_root.exists()


def test_delete_artifact_keeps_catalog_record_when_filesystem_delete_fails(
    tmp_path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "outputs"
    artifact_root = storage_root / "artifacts" / "ethereum" / "art_1"
    ensure_state_db(state_db_path(artifact_root), root_kind=RootKind.ARTIFACT, tables=())
    record = _artifact_record(artifact_root)
    upsert_catalog_record(storage_root, record)

    def fail_remove_path(path) -> None:
        del path
        raise OSError("delete failed")

    monkeypatch.setattr("spice.storage.lifecycle.remove_path", fail_remove_path)

    with pytest.raises(OSError, match="delete failed"):
        delete_catalog_artifact_root(storage_root, record)

    records = list_artifact_records(storage_root)
    assert [stored.artifact_id for stored in records] == [record.artifact_id]

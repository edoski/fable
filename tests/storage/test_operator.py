from __future__ import annotations

from pathlib import Path

from spice.storage.catalog.index import upsert_catalog_record
from spice.storage.operator import (
    StorageDeleteCommand,
    StorageDeleteFailure,
    StorageShowFailure,
    StorageShowQuery,
    StorageShowRendered,
    delete_storage,
    show_storage,
)
from spice.storage.selectors import ArtifactSelector, DatasetSelector
from tests.catalog_helpers import artifact_record, dataset_record, study_record


def _dataset_record(storage_root: Path):
    return dataset_record(
        storage_root / "corpora" / "ethereum" / "cor_1",
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
    )


def _study_record(storage_root: Path):
    return study_record(
        storage_root / "studies" / "ethereum" / "std_1",
        study_id="std_1",
        study_name="study",
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
    )


def _artifact_record(
    storage_root: Path,
    artifact_id: str,
    *,
    model_id: str = "model",
):
    return artifact_record(
        storage_root / "artifacts" / "ethereum" / artifact_id,
        artifact_id=artifact_id,
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id=model_id,
        problem_id="problem",
        variant="baseline",
        study_id="std_1",
        study_name="study",
    )


def _write_catalog_records(storage_root: Path, *records) -> None:
    for record in records:
        upsert_catalog_record(storage_root, record)


def test_show_detail_ambiguity_returns_diagnostics_and_narrowing_attributes(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "outputs"
    _write_catalog_records(
        storage_root,
        _artifact_record(storage_root, "art_1", model_id="lstm"),
        _artifact_record(storage_root, "art_2", model_id="transformer"),
    )

    outcome = show_storage(
        StorageShowQuery(
            storage_root=storage_root,
            kind="artifact",
            selector=ArtifactSelector(),
            detail="epochs",
        )
    )

    assert isinstance(outcome, StorageShowFailure)
    assert outcome.message == "--detail requires exactly one artifact match"
    assert outcome.diagnostics[0].title == "artifact matches"
    assert "model_id" in outcome.narrowing_attributes


def test_show_filtered_unique_match_renders_detail(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "outputs"
    record = _artifact_record(storage_root, "art_1", model_id="lstm")
    _write_catalog_records(
        storage_root,
        record,
        _artifact_record(storage_root, "art_2", model_id="transformer"),
    )
    seen: dict[str, object] = {}

    def fake_describe_root(root_path: Path, *, detail: str | None):
        seen.update({"root_path": root_path, "detail": detail})
        return object()

    monkeypatch.setattr("spice.storage.operator.describe_root", fake_describe_root)
    monkeypatch.setattr(
        "spice.storage.operator.sectioned_summary",
        lambda _description: ("artifact summary", [("artifact", [("artifact id", "art_1")])]),
    )

    outcome = show_storage(
        StorageShowQuery(
            storage_root=storage_root,
            kind="artifact",
            selector=ArtifactSelector(model_id="lstm"),
        )
    )

    assert isinstance(outcome, StorageShowRendered)
    assert seen == {"root_path": record.root_path, "detail": None}
    assert outcome.renderable.title == "artifact summary"


def test_delete_dataset_blocked_returns_dependent_sections(tmp_path: Path) -> None:
    storage_root = tmp_path / "outputs"
    _write_catalog_records(
        storage_root,
        _dataset_record(storage_root),
        _study_record(storage_root),
        _artifact_record(storage_root, "art_1"),
    )

    outcome = delete_storage(
        StorageDeleteCommand(
            storage_root=storage_root,
            kind="dataset",
            selector=DatasetSelector(dataset_id="cor_1"),
        )
    )

    assert isinstance(outcome, StorageDeleteFailure)
    assert outcome.message == "Dataset has dependent studies or artifacts."
    assert [diagnostic.title for diagnostic in outcome.diagnostics] == [
        "artifact matches",
        "study matches",
    ]

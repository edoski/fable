from __future__ import annotations

from pathlib import Path

from spice.storage.catalog import CatalogArtifactRecord, CatalogDatasetRecord, CatalogStudyRecord
from spice.storage.catalog.registry import (
    ARTIFACT_ROOT_SPEC,
    DATASET_ROOT_SPEC,
    STUDY_ROOT_SPEC,
)
from spice.storage.layout import catalog_db_path
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


def _dataset_record(storage_root: Path) -> CatalogDatasetRecord:
    root_path = storage_root / "corpora" / "ethereum" / "cor_1"
    return CatalogDatasetRecord(
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _study_record(storage_root: Path) -> CatalogStudyRecord:
    root_path = storage_root / "studies" / "ethereum" / "std_1"
    return CatalogStudyRecord(
        study_id="std_1",
        study_name="study",
        dataset_id="cor_1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _artifact_record(
    storage_root: Path,
    artifact_id: str,
    *,
    model_id: str = "model",
) -> CatalogArtifactRecord:
    root_path = storage_root / "artifacts" / "ethereum" / artifact_id
    return CatalogArtifactRecord(
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
        root_path=root_path,
        state_db_path=root_path / ".spice" / "state.sqlite",
    )


def _write_catalog_records(storage_root: Path, *records) -> None:
    catalog_path = catalog_db_path(storage_root)
    for record in records:
        if isinstance(record, CatalogDatasetRecord):
            DATASET_ROOT_SPEC.upsert(catalog_path, record)
        elif isinstance(record, CatalogStudyRecord):
            STUDY_ROOT_SPEC.upsert(catalog_path, record)
        elif isinstance(record, CatalogArtifactRecord):
            ARTIFACT_ROOT_SPEC.upsert(catalog_path, record)


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


def test_show_detail_uses_unique_filtered_match(tmp_path: Path, monkeypatch) -> None:
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
            detail="epochs",
            has_filters=True,
        )
    )

    assert isinstance(outcome, StorageShowRendered)
    assert seen == {"root_path": record.root_path, "detail": "epochs"}
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

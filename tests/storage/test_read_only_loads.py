from __future__ import annotations

from pathlib import Path

import pytest

from spice.core.errors import MissingStateError
from spice.storage.artifact import (
    list_evaluation_summaries,
    load_artifact_manifest,
    load_evaluation_summary,
    load_training_summary,
)
from spice.storage.corpus import list_acquire_runs, load_dataset_manifest
from spice.storage.engine import DATASET_ROOT_KIND, create_state_engine, ensure_state_db
from spice.storage.schema import DATASET_TABLES, dataset_manifest
from spice.storage.study_manifest import load_study_manifest, try_load_study_manifest


def test_study_reads_are_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "studies" / "ethereum" / "study-1" / ".spice" / "state.sqlite"

    assert try_load_study_manifest(db_path) is None
    with pytest.raises(MissingStateError, match="Missing study manifest"):
        load_study_manifest(db_path)
    assert not db_path.exists()


def test_dataset_reads_are_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "datasets" / "ethereum" / "dataset-1" / ".spice" / "state.sqlite"

    with pytest.raises(MissingStateError, match="Missing dataset manifest"):
        load_dataset_manifest(db_path)
    assert list_acquire_runs(db_path) == []
    assert not db_path.exists()


def test_dataset_manifest_loads_chain_runtime_from_named_chain_config(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "datasets" / "polygon" / "dataset-1" / ".spice" / "state.sqlite"
    window = {"start_timestamp": 1, "end_timestamp": 2}
    validation = {
        "status": "ok",
        "rows": 1,
        "block_range": {"first": 10, "last": 10},
        "timestamp_range": {"first": 1, "last": 1},
        "issues": None,
    }
    payload = {
        "dataset": {"id": "cor_test", "name": "icdcs_2026"},
        "chain": {"name": "polygon"},
        "request": {"history": window, "evaluation": window},
        "coverage": {"history": window, "evaluation": window},
        "validation": {"history": validation, "evaluation": validation},
    }

    ensure_state_db(db_path, root_kind=DATASET_ROOT_KIND, tables=DATASET_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            conn.execute(dataset_manifest.insert().values(singleton=1, payload=payload))
    finally:
        engine.dispose()

    manifest = load_dataset_manifest(db_path)

    assert manifest.chain.name == "polygon"
    assert manifest.chain.runtime.chain_id == 137
    assert manifest.chain.runtime.nominal_block_time_seconds == 2.0


def test_artifact_reads_are_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts" / "ethereum" / "artifact-1" / ".spice" / "state.sqlite"

    with pytest.raises(MissingStateError, match="Missing artifact manifest"):
        load_artifact_manifest(db_path)
    assert load_training_summary(db_path) is None
    assert load_evaluation_summary(db_path) is None
    assert list_evaluation_summaries(db_path) == []
    assert not db_path.exists()

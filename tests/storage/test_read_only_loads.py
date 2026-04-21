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


def test_dataset_reads_are_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "datasets" / "ethereum" / "dataset-1" / ".spice" / "state.sqlite"

    with pytest.raises(MissingStateError, match="Missing dataset manifest"):
        load_dataset_manifest(db_path)
    assert list_acquire_runs(db_path) == []
    assert not db_path.exists()


def test_artifact_reads_are_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts" / "ethereum" / "artifact-1" / ".spice" / "state.sqlite"

    with pytest.raises(MissingStateError, match="Missing artifact manifest"):
        load_artifact_manifest(db_path)
    assert load_training_summary(db_path) is None
    assert load_evaluation_summary(db_path) is None
    assert list_evaluation_summaries(db_path) == []
    assert not db_path.exists()

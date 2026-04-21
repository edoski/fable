from __future__ import annotations

from pathlib import Path

import pytest

from spice.core.errors import MissingStateError
from spice.storage.study_manifest import load_study_manifest, try_load_study_manifest


def test_try_load_study_manifest_is_read_only_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "studies" / "ethereum" / "study-1" / ".spice" / "state.sqlite"

    assert try_load_study_manifest(db_path) is None
    assert not db_path.exists()


def test_load_study_manifest_fails_cleanly_for_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "studies" / "ethereum" / "study-1" / ".spice" / "state.sqlite"

    with pytest.raises(MissingStateError, match="Missing study manifest"):
        load_study_manifest(db_path)
    assert not db_path.exists()

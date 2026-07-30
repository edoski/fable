from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from fable.experiments import (
    ExperimentKind,
    ExperimentManifest,
    experiment_manifest_path,
    load_experiment_manifest,
)

EXPERIMENT_ID = UUID("10000000-0000-4000-8000-000000000001")
OTHER_EXPERIMENT_ID = UUID("10000000-0000-4000-8000-000000000002")
RECORD_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_RECORD_ID = UUID("30000000-0000-4000-8000-000000000001")


def test_experiment_manifest_loads_ordered_typed_mapping_from_exact_directory(
    tmp_path: Path,
) -> None:
    manifest = ExperimentManifest(
        root={
            "ethereum.lstm": RECORD_ID,
            "polygon.transformer": OTHER_RECORD_ID,
        }
    )
    path = experiment_manifest_path(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID)
    path.parent.mkdir(parents=True)
    path.write_text(manifest.model_dump_json(), encoding="utf-8")

    loaded = load_experiment_manifest(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID)

    assert path == (
        tmp_path / "experiments" / "hpo" / str(EXPERIMENT_ID) / "manifest.json"
    )
    assert list(loaded.items()) == [
        ("ethereum.lstm", RECORD_ID),
        ("polygon.transformer", OTHER_RECORD_ID),
    ]


def test_experiment_manifest_loader_rejects_empty_mapping(tmp_path: Path) -> None:
    path = experiment_manifest_path(tmp_path, ExperimentKind.HPO, OTHER_EXPERIMENT_ID)
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match="at least 1 item"):
        load_experiment_manifest(tmp_path, ExperimentKind.HPO, OTHER_EXPERIMENT_ID)

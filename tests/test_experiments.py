from pathlib import Path
from uuid import UUID

import pytest

from fable.experiments import (
    ExperimentEntry,
    ExperimentKind,
    ExperimentManifest,
    experiment_manifest_path,
    load_experiment_manifest,
    write_experiment_manifest,
)

EXPERIMENT_ID = UUID("10000000-0000-4000-8000-000000000001")
OTHER_EXPERIMENT_ID = UUID("10000000-0000-4000-8000-000000000002")
RECORD_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_RECORD_ID = UUID("30000000-0000-4000-8000-000000000001")


def _manifest(
    *,
    record_id: UUID = RECORD_ID,
    cell: str = "ethereum/lstm/full",
) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(ExperimentEntry(cell=cell, record_id=record_id),),
    )


def test_experiment_manifest_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()

    write_experiment_manifest(tmp_path, ExperimentKind.HPO, manifest)

    assert load_experiment_manifest(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID) == manifest


def test_experiment_manifest_loader_rejects_wrong_requested_id(tmp_path: Path) -> None:
    manifest = _manifest()
    path = experiment_manifest_path(tmp_path, ExperimentKind.HPO, OTHER_EXPERIMENT_ID)
    path.parent.mkdir(parents=True)
    path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest ID does not match"):
        load_experiment_manifest(tmp_path, ExperimentKind.HPO, OTHER_EXPERIMENT_ID)


def test_experiment_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    original = _manifest()
    replacement = _manifest(record_id=OTHER_RECORD_ID, cell="ethereum/lstm/hpo")
    write_experiment_manifest(tmp_path, ExperimentKind.HPO, original)

    with pytest.raises(FileExistsError):
        write_experiment_manifest(tmp_path, ExperimentKind.HPO, replacement)

    assert load_experiment_manifest(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID) == original

import json
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
RECORD_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_RECORD_ID = UUID("30000000-0000-4000-8000-000000000001")


@pytest.mark.parametrize("kind", tuple(ExperimentKind))
def test_experiment_manifest_kind_round_trips_one_record_id(
    tmp_path: Path,
    kind: ExperimentKind,
) -> None:
    manifest = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(
            ExperimentEntry(
                cell="ethereum/lstm/full",
                record_id=RECORD_ID,
            ),
        ),
    )

    write_experiment_manifest(tmp_path, kind, manifest)

    loaded = load_experiment_manifest(
        tmp_path,
        kind,
        EXPERIMENT_ID,
    )
    assert loaded == manifest
    assert json.loads(
        experiment_manifest_path(
            tmp_path,
            kind,
            EXPERIMENT_ID,
        ).read_text(encoding="utf-8")
    ) == {
        "experiment_id": str(EXPERIMENT_ID),
        "entries": [
            {
                "cell": "ethereum/lstm/full",
                "record_id": str(RECORD_ID),
            },
        ],
    }


def test_experiment_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    original = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(ExperimentEntry(cell="ethereum/lstm/full", record_id=RECORD_ID),),
    )
    replacement = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(ExperimentEntry(cell="ethereum/lstm/hpo", record_id=OTHER_RECORD_ID),),
    )
    write_experiment_manifest(tmp_path, ExperimentKind.HPO, original)

    with pytest.raises(FileExistsError):
        write_experiment_manifest(tmp_path, ExperimentKind.HPO, replacement)

    assert load_experiment_manifest(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID) == original

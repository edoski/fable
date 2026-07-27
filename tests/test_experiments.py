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
ARTIFACT_ID = UUID("20000000-0000-4000-8000-000000000001")
STUDY_ID = UUID("30000000-0000-4000-8000-000000000001")
EVALUATION_ID = UUID("40000000-0000-4000-8000-000000000001")


def test_experiment_manifest_round_trip_preserves_cell_references(tmp_path: Path) -> None:
    manifest = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(
            ExperimentEntry(
                cell="ethereum/lstm/full",
                artifact_id=ARTIFACT_ID,
                evaluation_id=EVALUATION_ID,
            ),
            ExperimentEntry(cell="ethereum/lstm/hpo", study_id=STUDY_ID),
        ),
    )

    write_experiment_manifest(tmp_path, ExperimentKind.FEATURE_ABLATION, manifest)

    loaded = load_experiment_manifest(
        tmp_path,
        ExperimentKind.FEATURE_ABLATION,
        EXPERIMENT_ID,
    )
    assert loaded == manifest
    assert loaded.entries[0].require_artifact_id() == ARTIFACT_ID
    assert loaded.entries[0].require_evaluation_id() == EVALUATION_ID
    assert loaded.entries[1].require_study_id() == STUDY_ID
    assert json.loads(
        experiment_manifest_path(
            tmp_path,
            ExperimentKind.FEATURE_ABLATION,
            EXPERIMENT_ID,
        ).read_text(encoding="utf-8")
    ) == {
        "experiment_id": str(EXPERIMENT_ID),
        "entries": [
            {
                "cell": "ethereum/lstm/full",
                "artifact_id": str(ARTIFACT_ID),
                "evaluation_id": str(EVALUATION_ID),
            },
            {
                "cell": "ethereum/lstm/hpo",
                "study_id": str(STUDY_ID),
            },
        ],
    }


def test_experiment_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    original = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(ExperimentEntry(cell="ethereum/lstm/full", artifact_id=ARTIFACT_ID),),
    )
    replacement = ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        entries=(ExperimentEntry(cell="ethereum/lstm/hpo", study_id=STUDY_ID),),
    )
    write_experiment_manifest(tmp_path, ExperimentKind.HPO, original)

    with pytest.raises(FileExistsError):
        write_experiment_manifest(tmp_path, ExperimentKind.HPO, replacement)

    assert load_experiment_manifest(tmp_path, ExperimentKind.HPO, EXPERIMENT_ID) == original


def test_experiment_entry_requires_a_canonical_reference() -> None:
    with pytest.raises(ValueError, match="entry must reference a canonical record"):
        ExperimentEntry(cell="unresolved")

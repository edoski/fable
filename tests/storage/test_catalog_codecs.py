from __future__ import annotations

import json
from pathlib import Path

import pytest

from spice.core.errors import SpiceOperatorError
from spice.storage.catalog.codecs import (
    decode_remote_catalog_record,
    encode_remote_catalog_record,
)
from spice.storage.catalog.records import (
    CatalogArtifactRecord,
    CatalogDatasetRecord,
    CatalogStudyRecord,
)
from spice.storage.engine import RootKind


def _dataset_record() -> CatalogDatasetRecord:
    root = Path("/storage/corpora/ethereum/dataset-1")
    return CatalogDatasetRecord(
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        root_path=root,
        state_db_path=root / ".spice" / "state.sqlite",
    )


def _study_record() -> CatalogStudyRecord:
    root = Path("/storage/studies/ethereum/study-1")
    return CatalogStudyRecord(
        study_id="study-1",
        study_name="study",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        root_path=root,
        state_db_path=root / ".spice" / "state.sqlite",
    )


def _artifact_record(*, study_id: str | None = None) -> CatalogArtifactRecord:
    root = Path("/storage/artifacts/ethereum/artifact-1")
    return CatalogArtifactRecord(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="dataset",
        chain_name="ethereum",
        features_id="features",
        prediction_id="prediction",
        model_id="model",
        problem_id="problem",
        variant="baseline",
        study_id=study_id,
        study_name=None if study_id is None else "study",
        root_path=root,
        state_db_path=root / ".spice" / "state.sqlite",
    )


@pytest.mark.parametrize(
    ("record", "root_kind"),
    [
        (_dataset_record(), "corpus"),
        (_study_record(), "study"),
        (_artifact_record(), "artifact"),
        (_artifact_record(study_id="study-1"), "artifact"),
    ],
)
def test_remote_catalog_record_codec_round_trips_records(record, root_kind: str) -> None:
    encoded = encode_remote_catalog_record(record)
    decoded = decode_remote_catalog_record(encoded)

    assert json.loads(encoded)["root_kind"] == root_kind
    assert decoded == record


def test_remote_catalog_record_codec_rejects_kind_mismatch() -> None:
    with pytest.raises(SpiceOperatorError, match="root kind mismatch"):
        decode_remote_catalog_record(
            encode_remote_catalog_record(_dataset_record()),
            expected_root_kind=RootKind.ARTIFACT,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda record: record.pop("dataset_id"), "missing fields"),
        (lambda record: record.__setitem__("extra", "value"), "extra fields"),
        (lambda record: record.__setitem__("dataset_name", None), "cannot be null"),
        (lambda record: record.__setitem__("dataset_name", 1), "must be a string"),
    ],
)
def test_remote_catalog_record_codec_rejects_invalid_record_payload(mutate, match) -> None:
    payload = json.loads(encode_remote_catalog_record(_dataset_record()))
    mutate(payload["record"])

    with pytest.raises(SpiceOperatorError, match=match):
        decode_remote_catalog_record(json.dumps(payload))


def test_remote_catalog_record_codec_rejects_malformed_payload() -> None:
    with pytest.raises(SpiceOperatorError, match="not valid JSON"):
        decode_remote_catalog_record("{")

from __future__ import annotations

from typing import cast

import pytest

from spice.config import TuneConfig, WorkflowTask
from spice.core.errors import StateConflictError, StateLayoutError
from spice.corpus.metadata import (
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    CorpusAcquisitionSourceRequirements,
    CorpusSplitManifest,
    CorpusSplitManifests,
    DatasetIdentity,
    DatasetManifest,
    SplitCoverageMetadata,
    SplitMaterializationMetadata,
    SplitRequestMetadata,
    TimestampRangeMetadata,
)
from spice.storage.root_identity import produced_study_id
from spice.storage.study_manifest import (
    insert_study_manifest,
    load_study_manifest,
    manifest_from_tune_config,
    try_load_study_manifest,
)
from spice.storage.study_manifest_codecs import STUDY_MANIFEST_CODEC
from tests.root_handle_helpers import corpus_handle, study_handle

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _corpus_manifest(config: TuneConfig) -> DatasetManifest:
    split = CorpusSplitManifest(
        kind="history",
        request=SplitRequestMetadata(
            start_timestamp=1,
            end_timestamp=2,
            start_block=1,
            end_block=2,
        ),
        coverage=SplitCoverageMetadata(
            first_timestamp=1,
            last_timestamp=2,
            first_block=1,
            last_block=1,
            rows=1,
        ),
        validation=CompactValidationReport(
            status="clean",
            rows=1,
            block_range=BlockRangeMetadata(first=1, last=1),
            timestamp_range=TimestampRangeMetadata(first=1, last=2),
        ),
        materialization=SplitMaterializationMetadata(outcome="reused", file_count=1),
    )
    return DatasetManifest(
        dataset=DatasetIdentity(id=TEST_DATASET_ID, name=config.dataset.name),
        chain=ChainMetadata(name=config.chain.name, runtime=config.chain.runtime),
        splits=CorpusSplitManifests(history=split, evaluation=split),
        source_requirements=CorpusAcquisitionSourceRequirements(
            required_columns=frozenset(
                {"block_number", "timestamp", "chain_id", "base_fee_per_gas"}
            ),
            optional_enrichments=frozenset(),
            temporal_unit="block",
            ordering_key="block_number",
            partition_key="chain_id",
        ),
    )


def _study_manifest(tmp_path, load_workflow_config, *, study_name: str):
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            study=study_name,
        ),
    )
    corpus = corpus_handle(
        config.storage.root,
        chain_name=config.chain.name,
        dataset_id=TEST_DATASET_ID,
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        config.storage.root,
        corpus=corpus,
        study_id=produced_study_id(config),
        study_name=config.study.name,
    )
    return manifest_from_tune_config(
        config,
        corpus=corpus,
        study=study,
        corpus_manifest=_corpus_manifest(config),
    )


def test_study_manifest_codec_round_trips_canonical_definition(
    tmp_path,
    load_workflow_config,
) -> None:
    manifest = _study_manifest(tmp_path, load_workflow_config, study_name="roundtrip_probe")

    restored = STUDY_MANIFEST_CODEC.decode(STUDY_MANIFEST_CODEC.encode(manifest))

    assert restored == manifest


def test_study_manifest_codec_rejects_malformed_payload(
    tmp_path,
    load_workflow_config,
) -> None:
    with pytest.raises(StateLayoutError, match="Invalid study manifest payload"):
        STUDY_MANIFEST_CODEC.decode({"unexpected": 1})


def test_study_manifest_persists_once_and_loads(tmp_path, load_workflow_config) -> None:
    manifest = _study_manifest(tmp_path, load_workflow_config, study_name="persist_probe")
    db_path = tmp_path / ".spice" / "state.sqlite"

    insert_study_manifest(db_path, manifest=manifest)

    assert load_study_manifest(db_path) == manifest
    assert try_load_study_manifest(db_path) == manifest
    with pytest.raises(StateConflictError, match="Study manifest already exists"):
        insert_study_manifest(db_path, manifest=manifest)

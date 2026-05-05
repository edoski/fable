from __future__ import annotations

from typing import cast

import pytest

from spice.config import TuneConfig, WorkflowTask
from spice.core.errors import StateLayoutError
from spice.corpus.metadata import (
    BlockRangeMetadata,
    ChainMetadata,
    CompactValidationReport,
    CorpusSplitManifest,
    CorpusSplitManifests,
    DatasetIdentity,
    DatasetManifest,
    SplitCoverageMetadata,
    SplitMaterializationMetadata,
    SplitRequestMetadata,
    TimestampRangeMetadata,
)
from spice.storage.study_manifest import manifest_from_tune_config
from spice.storage.study_manifest_codecs import (
    study_manifest_from_payload,
    study_manifest_payload,
)
from spice.workflows.preparation import produced_study_id
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
    )


def test_study_manifest_round_trips_through_canonical_definition_payload(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            study="roundtrip_probe",
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
    manifest = manifest_from_tune_config(
        config,
        corpus=corpus,
        study=study,
        corpus_manifest=_corpus_manifest(config),
    )
    restored = study_manifest_from_payload(study_manifest_payload(manifest))

    assert restored == manifest


def test_study_manifest_payload_rejects_extra_keys_and_loose_scalars(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            study="strict_payload_probe",
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
    manifest = manifest_from_tune_config(
        config,
        corpus=corpus,
        study=study,
        corpus_manifest=_corpus_manifest(config),
    )
    payload = study_manifest_payload(manifest)
    payload["extra"] = "nope"

    with pytest.raises(StateLayoutError, match="Invalid study manifest payload"):
        study_manifest_from_payload(payload)

    payload = study_manifest_payload(manifest)
    definition = cast(dict[str, object], payload["definition"])
    definition["extra"] = "nope"

    with pytest.raises(StateLayoutError, match="Invalid study manifest payload"):
        study_manifest_from_payload(payload)

    payload = study_manifest_payload(manifest)
    payload["sampler_seed"] = "2026"

    with pytest.raises(StateLayoutError, match="Invalid study manifest payload"):
        study_manifest_from_payload(payload)

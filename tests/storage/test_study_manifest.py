from __future__ import annotations

from typing import cast

from spice.config import TuneConfig, WorkflowTask
from spice.corpus.metadata import (
    ChainMetadata,
    DatasetCoverageMetadata,
    DatasetIdentity,
    DatasetManifest,
    DatasetRequestMetadata,
    DatasetValidationMetadata,
    DatasetWindowMetadata,
)
from spice.storage.study_manifest import (
    manifest_from_payload,
    manifest_from_tune_config,
    manifest_payload,
)
from spice.storage.workflow_roots import produced_study_id
from tests.root_handle_helpers import corpus_handle, study_handle

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _corpus_manifest(config: TuneConfig) -> DatasetManifest:
    window = DatasetWindowMetadata(start_timestamp=1, end_timestamp=2)
    return DatasetManifest(
        dataset=DatasetIdentity(id=TEST_DATASET_ID, name=config.dataset.name),
        chain=ChainMetadata(name=config.chain.name, runtime=config.chain.runtime),
        request=DatasetRequestMetadata(history=window, evaluation=window),
        coverage=DatasetCoverageMetadata(history=window, evaluation=window),
        validation=DatasetValidationMetadata(history=None, evaluation=None),
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
    restored = manifest_from_payload(manifest_payload(manifest))

    assert restored == manifest

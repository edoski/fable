from __future__ import annotations

from typing import cast

import yaml

from spice.config import (
    AcquireConfig,
    AcquireWorkflowSelection,
    TrainConfig,
    TrainWorkflowSelection,
    TuneConfig,
    TuneWorkflowSelection,
    resolve_workflow_config,
)
from spice.config.groups import load_named_group_payload
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
from spice.storage.identity import (
    study_definition_identity_from_manifest,
    study_definition_identity_from_tuned_config,
)
from spice.storage.root_identity import (
    produced_artifact_id,
    produced_corpus_id,
    produced_study_id,
)
from spice.storage.study_manifest import manifest_from_tune_config
from tests.root_handle_helpers import corpus_handle, study_handle

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"


def _write_surface(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "surface" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_surface(conf_root) -> dict[str, object]:
    return load_named_group_payload("current_row_fee_dynamics", "surface")


def _tune_config(tmp_path, *, surface: str, objective: str | None = None) -> TuneConfig:
    return cast(
        TuneConfig,
        resolve_workflow_config(
            TuneWorkflowSelection(
                surface=surface,
                dataset_id=TEST_DATASET_ID,
                objective=objective,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )


def _train_config(
    tmp_path,
    *,
    surface: str,
    objective: str | None = None,
    study_id: str | None = None,
) -> TrainConfig:
    return cast(
        TrainConfig,
        resolve_workflow_config(
            TrainWorkflowSelection(
                surface=surface,
                dataset_id=None if study_id is not None else TEST_DATASET_ID,
                study_id=study_id,
                variant="tuned" if study_id is not None else "baseline",
                objective=objective,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )


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


def test_study_id_uses_resolved_identity_but_not_trial_limits(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    payload = {
        **_base_surface(conf_root),
        "tuning": {"id": "default", "space": "lstm_large_capacity"},
    }
    _write_surface(conf_root, "study_identity_change", payload)

    base = _tune_config(tmp_path, surface="current_row_fee_dynamics")
    changed = _tune_config(tmp_path, surface="study_identity_change")
    limited = cast(
        TuneConfig,
        resolve_workflow_config(
            TuneWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                trial_count=40,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    evaluation_path = conf_root / "evaluation" / "poisson_replay.yaml"
    evaluation_payload = load_named_group_payload("poisson_replay", "evaluation")
    evaluation_payload["seed"] = 3030
    evaluation_path.write_text(
        yaml.safe_dump(evaluation_payload, sort_keys=False),
        encoding="utf-8",
    )
    changed_evaluation = _tune_config(tmp_path, surface="current_row_fee_dynamics")

    assert produced_study_id(changed) != produced_study_id(base)
    assert produced_study_id(changed_evaluation) != produced_study_id(base)
    assert produced_study_id(limited) == produced_study_id(base)


def test_artifact_id_uses_training_identity_and_selected_dataset(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    _write_surface(
        conf_root,
        "artifact_identity_change",
        {**_base_surface(conf_root), "problem": "current_row_recent_median"},
    )

    base = _train_config(tmp_path, surface="current_row_fee_dynamics")
    changed_problem = _train_config(tmp_path, surface="artifact_identity_change")
    changed_objective = _train_config(
        tmp_path,
        surface="current_row_fee_dynamics",
        objective="validation_total_loss",
    )
    evaluation_path = conf_root / "evaluation" / "poisson_replay.yaml"
    evaluation_payload = load_named_group_payload("poisson_replay", "evaluation")
    evaluation_payload["seed"] = 3030
    evaluation_path.write_text(
        yaml.safe_dump(evaluation_payload, sort_keys=False),
        encoding="utf-8",
    )
    changed_evaluation = _train_config(tmp_path, surface="current_row_fee_dynamics")

    assert produced_artifact_id(changed_problem, dataset_id=TEST_DATASET_ID) != (
        produced_artifact_id(base, dataset_id=TEST_DATASET_ID)
    )
    assert produced_artifact_id(changed_objective, dataset_id=TEST_DATASET_ID) != (
        produced_artifact_id(base, dataset_id=TEST_DATASET_ID)
    )
    assert produced_artifact_id(changed_evaluation, dataset_id=TEST_DATASET_ID) != (
        produced_artifact_id(base, dataset_id=TEST_DATASET_ID)
    )


def test_storage_root_does_not_affect_produced_ids(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()
    first = _train_config(tmp_path / "one", surface="current_row_fee_dynamics")
    second = _train_config(tmp_path / "two", surface="current_row_fee_dynamics")

    assert produced_artifact_id(first, dataset_id=TEST_DATASET_ID) == (
        produced_artifact_id(second, dataset_id=TEST_DATASET_ID)
    )


def test_tuned_definition_identity_matches_stored_study_manifest(
    tmp_path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()
    tune_config = _tune_config(tmp_path, surface="current_row_fee_dynamics")
    study_id = produced_study_id(tune_config)
    train_config = _train_config(
        tmp_path,
        surface="current_row_fee_dynamics",
        study_id=study_id,
    )
    corpus = corpus_handle(
        tune_config.storage.root,
        chain_name=tune_config.chain.name,
        dataset_id=TEST_DATASET_ID,
        dataset_name=tune_config.dataset.name,
    )
    study = study_handle(
        tune_config.storage.root,
        corpus=corpus,
        study_id=study_id,
        study_name=tune_config.study.name,
    )
    manifest = manifest_from_tune_config(
        tune_config,
        corpus=corpus,
        study=study,
        corpus_manifest=_corpus_manifest(tune_config),
    )

    assert study_definition_identity_from_manifest(manifest) == (
        study_definition_identity_from_tuned_config(
            train_config,
            study_id=study_id,
            chain_name=manifest.chain_name,
            dataset_id=manifest.dataset_id,
            dataset_name=manifest.dataset_name,
        )
    )


def test_corpus_id_uses_dataset_evaluation_date(tmp_path, isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    base = cast(
        AcquireConfig,
        resolve_workflow_config(
            AcquireWorkflowSelection(
                surface="current_row_fee_dynamics",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    changed_dataset = dict(load_named_group_payload("icdcs_2026", "dataset"))
    changed_dataset["evaluation_date"] = "2025-11-10"
    (conf_root / "dataset" / "icdcs_2026.yaml").write_text(
        yaml.safe_dump(changed_dataset, sort_keys=False),
        encoding="utf-8",
    )
    changed = cast(
        AcquireConfig,
        resolve_workflow_config(
            AcquireWorkflowSelection(
                surface="current_row_fee_dynamics",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert produced_corpus_id(changed) != produced_corpus_id(base)

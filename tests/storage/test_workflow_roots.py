from __future__ import annotations

from typing import cast

from spice.config import EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.config.models import ArtifactVariant
from spice.storage.catalog.records import (
    CatalogArtifactRecord,
    CatalogDatasetRecord,
    CatalogStudyRecord,
)
from spice.storage.workflow_roots import (
    BaselineTrainWorkflowRoots,
    TunedTrainWorkflowRoots,
    artifact_root_from_record,
    corpus_root_from_record,
    study_root_from_record,
)
from spice.workflows.preparation import (
    produced_artifact_id,
    produced_study_id,
    resolve_evaluate_roots,
    resolve_train_roots,
    resolve_tune_roots,
)


def _dataset_record(tmp_path, *, dataset_id: str, chain_name: str = "ethereum"):
    root = tmp_path / "catalog" / "datasets" / dataset_id
    return CatalogDatasetRecord(
        dataset_id=dataset_id,
        dataset_name=f"{chain_name}_dataset",
        chain_name=chain_name,
        root_path=root,
        state_db_path=root / "custom-state.sqlite",
    )


def _study_record(tmp_path, *, dataset_id: str, chain_name: str = "ethereum"):
    root = tmp_path / "catalog" / "studies" / "std_existing"
    return CatalogStudyRecord(
        study_id="std_existing",
        study_name="existing_study",
        dataset_id=dataset_id,
        dataset_name=f"{chain_name}_dataset",
        chain_name=chain_name,
        features_id="core_fee_dynamics",
        prediction_id="icdcs_2026",
        model_id="lstm",
        problem_id="current_row_nominal",
        root_path=root,
        state_db_path=root / "custom-state.sqlite",
    )


def _artifact_record(
    tmp_path,
    *,
    artifact_id: str = "art_existing",
    dataset_id: str,
    chain_name: str = "ethereum",
):
    root = tmp_path / "catalog" / "artifacts" / artifact_id
    return CatalogArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        dataset_name=f"{chain_name}_dataset",
        chain_name=chain_name,
        features_id="core_fee_dynamics",
        prediction_id="icdcs_2026",
        model_id="lstm",
        problem_id="current_row_nominal",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=root,
        state_db_path=root / "custom-state.sqlite",
    )


def test_tune_consumer_roots_resolve_dataset_and_produced_study(
    tmp_path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = cast(
        TuneConfig,
        load_workflow_config(WorkflowTask.TUNE, workspace=tmp_path),
    )
    dataset = _dataset_record(tmp_path, dataset_id=config.dataset_id, chain_name="polygon")
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = resolve_tune_roots(config)

    assert roots.corpus.dataset_id == config.dataset_id
    assert roots.corpus.state_db_path == dataset.state_db_path
    assert roots.study.study_id == produced_study_id(config)
    assert roots.study.dataset_id == dataset.dataset_id
    assert roots.study.chain_name == "polygon"


def test_baseline_train_consumer_roots_resolve_dataset_and_produced_artifact(
    tmp_path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(WorkflowTask.TRAIN, workspace=tmp_path),
    )
    assert config.dataset_id is not None
    dataset = _dataset_record(tmp_path, dataset_id=config.dataset_id, chain_name="polygon")
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = resolve_train_roots(config)

    assert isinstance(roots, BaselineTrainWorkflowRoots)
    assert roots.corpus.state_db_path == dataset.state_db_path
    assert roots.artifact.artifact_id == produced_artifact_id(
        config,
        dataset_id=dataset.dataset_id,
    )
    assert roots.artifact.variant is ArtifactVariant.BASELINE


def test_tuned_train_consumer_roots_use_study_dataset_for_artifact_identity(
    tmp_path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            variant="tuned",
        ),
    )
    study = _study_record(tmp_path, dataset_id="cor_from_study", chain_name="polygon")
    dataset = _dataset_record(tmp_path, dataset_id=study.dataset_id, chain_name="polygon")
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_study_record",
        lambda *_args, **_kwargs: study,
    )
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = resolve_train_roots(config)

    assert isinstance(roots, TunedTrainWorkflowRoots)
    assert roots.study.state_db_path == study.state_db_path
    assert roots.corpus.dataset_id == "cor_from_study"
    assert roots.artifact.artifact_id == produced_artifact_id(
        config,
        dataset_id="cor_from_study",
    )
    assert roots.artifact.study_id == study.study_id


def test_evaluate_consumer_roots_resolve_dataset_and_artifact_independently(
    tmp_path,
    monkeypatch,
    load_workflow_config,
) -> None:
    config = cast(
        EvaluateConfig,
        load_workflow_config(WorkflowTask.EVALUATE, workspace=tmp_path),
    )
    dataset = _dataset_record(tmp_path, dataset_id=config.dataset_id, chain_name="polygon")
    artifact = _artifact_record(
        tmp_path,
        artifact_id=config.artifact_id,
        dataset_id="cor_artifact",
        chain_name="polygon",
    )
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "spice.workflows.preparation.resolve_artifact_record",
        lambda *_args, **_kwargs: artifact,
    )

    roots = resolve_evaluate_roots(config)

    assert roots.corpus.dataset_id == config.dataset_id
    assert roots.artifact.artifact_id == config.artifact_id
    assert roots.corpus.state_db_path == dataset.state_db_path
    assert roots.artifact.state_db_path == artifact.state_db_path
    assert roots.artifact.dataset_id == "cor_artifact"


def test_corpus_root_handle_loads_manifest(tmp_path, monkeypatch) -> None:
    dataset = _dataset_record(tmp_path, dataset_id="cor_existing")
    root = corpus_root_from_record(tmp_path, dataset)
    manifest = object()
    calls: list[object] = []

    monkeypatch.setattr(
        "spice.storage.workflow_roots.load_dataset_manifest",
        lambda db_path: calls.append(db_path) or manifest,
    )

    assert root.load_manifest() is manifest
    assert calls == [dataset.state_db_path]


def test_study_root_from_record_preserves_catalog_paths(tmp_path) -> None:
    study = _study_record(tmp_path, dataset_id="cor_existing")
    root = study_root_from_record(tmp_path, study)

    assert root.study_id == study.study_id
    assert root.root_path == study.root_path
    assert root.state_db_path == study.state_db_path


def test_artifact_root_from_record_preserves_catalog_paths(tmp_path) -> None:
    artifact = _artifact_record(tmp_path, dataset_id="cor_existing")
    root = artifact_root_from_record(tmp_path, artifact)

    assert root.artifact_id == artifact.artifact_id
    assert root.root_path == artifact.root_path
    assert root.state_db_path == artifact.state_db_path

from __future__ import annotations

from typing import cast

from spice.config import AcquireConfig, EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.config.models import ArtifactVariant
from spice.storage.root_identity import (
    produced_artifact_id,
    produced_corpus_id,
    produced_study_id,
)
from spice.storage.workflow_root_materialization import (
    materialize_acquire_roots,
    materialize_evaluate_roots,
    materialize_train_roots,
    materialize_tune_roots,
)
from spice.storage.workflow_roots import (
    BaselineTrainWorkflowRoots,
    CorpusRootHandle,
    TunedTrainWorkflowRoots,
)
from tests.catalog_helpers import artifact_record, dataset_record, study_record


def _dataset_record(tmp_path, *, dataset_id: str, chain_name: str = "ethereum"):
    root = tmp_path / "catalog" / "datasets" / dataset_id
    return dataset_record(
        root,
        dataset_id=dataset_id,
        dataset_name=f"{chain_name}_dataset",
        chain_name=chain_name,
        state_db=root / "custom-state.sqlite",
    )


def _study_record(tmp_path, *, dataset_id: str, chain_name: str = "ethereum"):
    root = tmp_path / "catalog" / "studies" / "std_existing"
    return study_record(
        root,
        study_id="std_existing",
        study_name="existing_study",
        dataset_id=dataset_id,
        dataset_name=f"{chain_name}_dataset",
        chain_name=chain_name,
        features_id="core_fee_dynamics",
        prediction_id="icdcs_2026",
        model_id="lstm",
        problem_id="current_row_nominal",
        state_db=root / "custom-state.sqlite",
    )


def _artifact_record(
    tmp_path,
    *,
    artifact_id: str = "art_existing",
    dataset_id: str,
    chain_name: str = "ethereum",
):
    root = tmp_path / "catalog" / "artifacts" / artifact_id
    return artifact_record(
        root,
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
        state_db=root / "custom-state.sqlite",
    )


def test_acquire_producer_roots_use_produced_corpus_identity(
    tmp_path,
    load_workflow_config,
) -> None:
    config = cast(
        AcquireConfig,
        load_workflow_config(WorkflowTask.ACQUIRE, workspace=tmp_path),
    )

    roots = materialize_acquire_roots(config)

    assert roots.corpus.dataset_id == produced_corpus_id(config)
    assert roots.corpus.dataset_name == config.dataset.name
    assert roots.corpus.chain_name == config.chain.name


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
        "spice.storage.workflow_root_materialization.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = materialize_tune_roots(config)

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
        "spice.storage.workflow_root_materialization.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = materialize_train_roots(config)

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
        "spice.storage.workflow_root_materialization.resolve_study_record",
        lambda *_args, **_kwargs: study,
    )
    monkeypatch.setattr(
        "spice.storage.workflow_root_materialization.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )

    roots = materialize_train_roots(config)

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
        "spice.storage.workflow_root_materialization.resolve_dataset_record",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "spice.storage.workflow_root_materialization.resolve_artifact_record",
        lambda *_args, **_kwargs: artifact,
    )

    roots = materialize_evaluate_roots(config)

    assert roots.corpus.dataset_id == config.dataset_id
    assert roots.artifact.artifact_id == config.artifact_id
    assert roots.corpus.state_db_path == dataset.state_db_path
    assert roots.artifact.state_db_path == artifact.state_db_path
    assert roots.artifact.dataset_id == "cor_artifact"


def test_corpus_root_handle_loads_manifest(tmp_path, monkeypatch) -> None:
    dataset = _dataset_record(tmp_path, dataset_id="cor_existing")
    root = CorpusRootHandle(
        storage_root=tmp_path,
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        chain_name=dataset.chain_name,
        root_path=dataset.root_path,
        state_db_path=dataset.state_db_path,
        history_dir=dataset.root_path / "history",
        evaluation_dir=dataset.root_path / "evaluation",
    )
    manifest = object()
    calls: list[object] = []

    monkeypatch.setattr(
        "spice.storage.workflow_roots.load_dataset_manifest",
        lambda db_path: calls.append(db_path) or manifest,
    )

    assert root.load_manifest() is manifest
    assert calls == [dataset.state_db_path]

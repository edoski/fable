from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from spice.config import TrainConfig, TuneConfig, WorkflowTask
from spice.storage.workflow_roots import CorpusRootHandle
from spice.workflows import _active_config, _training_preflight, preparation
from tests.root_handle_helpers import (
    baseline_train_roots,
    corpus_handle,
    study_handle,
    tune_roots,
    tuned_train_roots,
)


def test_train_preparation_uses_resolved_corpus_manifest(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=model_workflow_override(),
        ),
    )
    roots = baseline_train_roots(
        tmp_path / "outputs",
        corpus=corpus_handle(
            tmp_path / "outputs",
            chain_name="polygon",
            dataset_id=config.dataset_id,
            dataset_name="polygon_dataset",
        ),
    )
    corpus_manifest = SimpleNamespace(
        chain=SimpleNamespace(name="polygon", runtime=SimpleNamespace()),
        dataset=SimpleNamespace(name="polygon_dataset"),
    )
    captured: dict[str, object] = {}

    def fake_build_training_spec(active_config, *, corpus_manifest, **kwargs):
        captured["chain"] = active_config.chain.name
        captured["manifest_chain"] = corpus_manifest.chain.name
        captured["artifact_root"] = kwargs["artifact"].root_path
        return SimpleNamespace(
            training=SimpleNamespace(max_epochs=1),
            prediction_contract=SimpleNamespace(primary_metric_id="total_loss"),
            problem_contract=object(),
            feature_contract=object(),
        )

    monkeypatch.setattr(preparation, "materialize_train_roots", lambda _config: roots)
    monkeypatch.setattr(CorpusRootHandle, "load_manifest", lambda _self: corpus_manifest)
    monkeypatch.setattr(
        _training_preflight,
        "build_artifact_training_spec",
        fake_build_training_spec,
    )
    monkeypatch.setattr(
        _training_preflight,
        "training_coverage_requirement",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        _training_preflight,
        "validate_corpus_coverage",
        lambda *_args, **_kwargs: None,
    )

    prepared = preparation.prepare_train(config)

    assert prepared.roots is roots
    assert captured == {
        "chain": "ethereum",
        "manifest_chain": "polygon",
        "artifact_root": roots.artifact.root_path,
    }


def test_tuned_train_preparation_keeps_artifact_root_stable_after_best_params(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=model_workflow_override(),
            variant="tuned",
        ),
    )
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id="cor_9a73b1e88edb488afb1e",
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        tmp_path / "outputs",
        corpus=corpus,
        study_id="std_test",
        study_name=config.study.name,
    )
    roots = tuned_train_roots(
        tmp_path / "outputs",
        corpus=corpus,
        study=study,
        artifact_id="art_original",
    )
    tuned_training = config.training.model_copy(
        update={"batch_size": config.training.batch_size + 7}
    )
    tuned_config = config.model_copy(update={"training": tuned_training})
    captured: dict[str, object] = {}

    def fake_build_training_spec(active_config, *, artifact, **_kwargs):
        captured["batch_size"] = active_config.training.batch_size
        captured["artifact_root"] = artifact.root_path
        return SimpleNamespace(
            training=SimpleNamespace(max_epochs=1),
            prediction_contract=SimpleNamespace(primary_metric_id="total_loss"),
            problem_contract=object(),
            feature_contract=object(),
        )

    monkeypatch.setattr(preparation, "materialize_train_roots", lambda _config: roots)
    monkeypatch.setattr(
        _active_config,
        "apply_study_best_params",
        lambda *_args, **_kwargs: SimpleNamespace(config=tuned_config),
    )
    monkeypatch.setattr(
        CorpusRootHandle,
        "load_manifest",
        lambda _self: SimpleNamespace(
            chain=SimpleNamespace(name=config.chain.name, runtime=SimpleNamespace()),
            dataset=SimpleNamespace(name=config.dataset.name),
        ),
    )
    monkeypatch.setattr(
        _training_preflight,
        "build_artifact_training_spec",
        fake_build_training_spec,
    )
    monkeypatch.setattr(
        _training_preflight,
        "training_coverage_requirement",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        _training_preflight,
        "validate_corpus_coverage",
        lambda *_args, **_kwargs: None,
    )

    prepared = preparation.prepare_train(config)

    assert prepared.active_config is tuned_config
    assert captured == {
        "batch_size": tuned_training.batch_size,
        "artifact_root": roots.artifact.root_path,
    }


def test_tune_preparation_uses_resolved_corpus_manifest(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    override = model_workflow_override() | tune_override()
    config = cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name="polygon",
        dataset_id=config.dataset_id,
        dataset_name="polygon_dataset",
    )
    roots = tune_roots(
        tmp_path / "outputs",
        corpus=corpus,
        study=study_handle(
            tmp_path / "outputs",
            corpus=corpus,
            study_id="std_test",
            study_name=config.study.name,
        ),
    )
    corpus_manifest = SimpleNamespace(
        chain=SimpleNamespace(name="polygon", runtime=SimpleNamespace()),
        dataset=SimpleNamespace(name="polygon_dataset"),
    )
    captured: dict[str, object] = {}

    def fake_coverage_spec(active_config, *, corpus_manifest, **_kwargs):
        captured["chain"] = active_config.chain.name
        captured["manifest_chain"] = corpus_manifest.chain.name
        return SimpleNamespace(problem_contract=object(), feature_contract=object())

    monkeypatch.setattr(preparation, "materialize_tune_roots", lambda _config: roots)
    monkeypatch.setattr(CorpusRootHandle, "load_manifest", lambda _self: corpus_manifest)
    monkeypatch.setattr(_training_preflight, "build_tuning_coverage_spec", fake_coverage_spec)
    monkeypatch.setattr(
        _training_preflight,
        "training_coverage_requirement",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        _training_preflight,
        "validate_corpus_coverage",
        lambda *_args, **_kwargs: None,
    )

    prepared = preparation.prepare_tune(config)

    assert prepared.roots is roots
    assert captured == {"chain": "ethereum", "manifest_chain": "polygon"}

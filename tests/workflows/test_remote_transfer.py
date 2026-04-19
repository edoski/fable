from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

from spice.core.errors import StateConflictError
from spice.core.reporting import NullReporter
from spice.remote.transfer import pull_artifact_from_remote, push_study_to_remote
from spice.storage.layout import resolve_workflow_paths
from spice.storage.roots import ArtifactSelector, StudySelector, list_artifact_records
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune


def test_push_study_to_remote_uses_canonical_destination_root(
    tmp_path,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
    monkeypatch,
) -> None:
    config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    paths = resolve_workflow_paths(config)
    seed_history_dataset(config)
    run_tune(config, reporter=NullReporter())

    remote_storage_root = tmp_path / "remote-storage"
    captured: dict[str, object] = {}
    target = SimpleNamespace(
        spec=SimpleNamespace(
            paths=SimpleNamespace(storage_root=remote_storage_root),
        )
    )

    monkeypatch.setattr("spice.remote.transfer.resolve_remote_target", lambda: target)
    monkeypatch.setattr(
        "spice.remote.transfer._prepare_remote_stage",
        lambda _target, *, destination_root, staged_root, replace: captured.update(
            {
                "destination_root": destination_root,
                "staged_root": staged_root,
                "replace": replace,
            }
        ),
    )
    monkeypatch.setattr("spice.remote.transfer.run_rsync_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "spice.remote.transfer._finalize_remote_stage",
        lambda *_args, **_kwargs: None,
    )

    record = push_study_to_remote(
        storage_root=tmp_path / "outputs",
        selector=StudySelector(
            chain_name=config.chain.name,
            dataset_name=config.dataset.name,
            feature_set_id=config.feature_set.id,
            prediction_id=config.prediction.id,
            model_id=config.model.id,
            problem_id=config.problem.id,
            study_name=config.study.name,
        ),
        replace=False,
    )

    assert record.study_id == paths.study_id
    assert captured["destination_root"] == (
        remote_storage_root / "studies" / config.chain.name / paths.study_id
    )
    assert captured["replace"] is False


def test_pull_artifact_from_remote_reindexes_local_storage(
    tmp_path,
    load_test_train_config,
    model_workflow_override,
    seed_history_dataset,
    monkeypatch,
) -> None:
    remote_workspace = tmp_path / "remote-workspace"
    remote_config = load_test_train_config(
        remote_workspace,
        override=model_workflow_override(),
    )
    seed_history_dataset(remote_config)
    run_train(remote_config, reporter=NullReporter())
    remote_record = list_artifact_records(remote_workspace / "outputs")[0]

    target = SimpleNamespace(
        spec=SimpleNamespace(
            paths=SimpleNamespace(storage_root=remote_workspace / "outputs"),
        )
    )
    monkeypatch.setattr("spice.remote.transfer.resolve_remote_target", lambda: target)
    monkeypatch.setattr(
        "spice.remote.transfer._resolve_remote_artifact_record",
        lambda _target, *, selector: remote_record,
    )

    def fake_rsync_from_remote(_target, *, source_root, destination_root) -> None:
        shutil.copytree(source_root, destination_root, dirs_exist_ok=True)

    monkeypatch.setattr("spice.remote.transfer.run_rsync_from_remote", fake_rsync_from_remote)

    local_storage_root = tmp_path / "local-outputs"
    pulled_record, dataset_present = pull_artifact_from_remote(
        storage_root=local_storage_root,
        selector=ArtifactSelector(
            chain_name=remote_config.chain.name,
            dataset_name=remote_config.dataset.name,
            feature_set_id=remote_config.feature_set.id,
            prediction_id=remote_config.prediction.id,
            model_id=remote_config.model.id,
            problem_id=remote_config.problem.id,
            variant=remote_config.artifact.variant.value,
        ),
        replace=False,
    )

    assert pulled_record.artifact_id == remote_record.artifact_id
    assert dataset_present is False
    local_records = list_artifact_records(local_storage_root)
    assert len(local_records) == 1
    assert local_records[0].artifact_id == remote_record.artifact_id


def test_pull_artifact_from_remote_rejects_existing_destination(
    tmp_path,
    monkeypatch,
) -> None:
    remote_record = SimpleNamespace(
        artifact_id="artifact-1",
        dataset_id="dataset-1",
        dataset_name="icdcs_2026",
        chain_name="ethereum",
        feature_set_id="icdcs_2026",
        prediction_id="candidate_offset_selection",
        model_id="lstm",
        problem_id="icdcs_2026",
        variant="baseline",
        study_id=None,
        study_name=None,
        root_path=tmp_path / "remote-artifact",
        state_db_path=tmp_path / "remote-artifact" / ".spice" / "state.sqlite",
    )
    destination_root = tmp_path / "outputs" / "artifacts" / "ethereum" / "artifact-1"
    destination_root.mkdir(parents=True)

    monkeypatch.setattr(
        "spice.remote.transfer.resolve_remote_target",
        lambda: SimpleNamespace(spec=SimpleNamespace(paths=SimpleNamespace(storage_root=tmp_path))),
    )
    monkeypatch.setattr(
        "spice.remote.transfer._resolve_remote_artifact_record",
        lambda _target, *, selector: remote_record,
    )

    with pytest.raises(StateConflictError, match="Destination already exists"):
        pull_artifact_from_remote(
            storage_root=tmp_path / "outputs",
            selector=ArtifactSelector(),
            replace=False,
        )

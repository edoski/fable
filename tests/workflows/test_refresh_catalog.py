from __future__ import annotations

from spice.core.reporting import NullReporter
from spice.storage.layout import resolve_workflow_paths
from spice.storage.roots import list_artifact_records, list_study_records, refresh_catalog
from spice.workflows.train import run as run_train
from spice.workflows.tune import run as run_tune


def test_refresh_catalog_rebuilds_from_root_local_state(
    tmp_path,
    load_test_train_config,
    load_test_tune_config,
    model_workflow_override,
    seed_history_dataset,
    tune_override,
) -> None:
    tune_config = load_test_tune_config(
        tmp_path,
        override=model_workflow_override() | tune_override(),
    )
    seed_history_dataset(tune_config)
    run_tune(tune_config, reporter=NullReporter())

    train_config = load_test_train_config(
        tmp_path,
        override=model_workflow_override(),
    )
    seed_history_dataset(train_config)
    run_train(train_config, reporter=NullReporter())

    resolve_workflow_paths(train_config).catalog_db.unlink()

    summary = refresh_catalog(tmp_path / "outputs")

    assert summary.dataset_roots == 0
    assert summary.study_roots == 1
    assert summary.artifact_roots == 1

    studies = list_study_records(tmp_path / "outputs")
    artifacts = list_artifact_records(tmp_path / "outputs")
    assert len(studies) == 1
    assert len(artifacts) == 1
    assert studies[0].prediction_id == tune_config.prediction.id
    assert artifacts[0].prediction_id == train_config.prediction.id

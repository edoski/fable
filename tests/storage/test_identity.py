from __future__ import annotations

from typing import cast

import yaml

from spice.config import (
    EvaluateConfig,
    EvaluateWorkflowRequest,
    TrainConfig,
    TrainWorkflowRequest,
    TuneConfig,
    TunedParameterSet,
    TuneWorkflowRequest,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.models import TunedTrainingParams
from spice.config.registry import load_named_group
from spice.modeling.pipeline import build_training_spec
from spice.modeling.tuning import apply_tuned_parameters
from spice.storage.identity import (
    study_request_identity_from_manifest,
    study_request_identity_from_tuned_config,
)
from spice.storage.layout import resolve_workflow_paths
from spice.storage.study_manifest import manifest_from_tune_config


def _write_surface(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "surface" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_surface(conf_root) -> dict[str, object]:
    return load_named_group("same_block_closed", "surface")


def _updated_surface(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    return {**base, **override}


def _train_paths(
    tmp_path,
    *,
    surface: str,
    variant: str | None = None,
    objective: str | None = None,
):
    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(
                surface=surface,
                variant=variant,
                objective=objective,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    return resolve_workflow_paths(config)


def _tune_paths(
    tmp_path,
    *,
    surface: str,
    objective: str | None = None,
    trial_count: int | None = None,
):
    config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowRequest(
                surface=surface,
                objective=objective,
                trial_count=trial_count,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    return resolve_workflow_paths(config)


def _evaluate_paths(tmp_path, *, surface: str, objective: str | None = None):
    config = cast(
        EvaluateConfig,
        resolve_workflow_config(
            WorkflowTask.EVALUATE,
            EvaluateWorkflowRequest(
                surface=surface,
                objective=objective,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    return resolve_workflow_paths(config)


def test_study_id_uses_full_resolved_identity(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    name = "study_identity_change"
    override = {"tuning_space": "lstm_extensive_calibrated"}
    payload = _updated_surface(_base_surface(conf_root), override)
    _write_surface(conf_root, name, payload)

    base = _tune_paths(tmp_path, surface="same_block_closed")
    changed = _tune_paths(tmp_path, surface=name)

    assert base.study_id is not None
    assert changed.study_id is not None
    assert changed.study_id != base.study_id


def test_study_id_ignores_tuning_run_limits(
    tmp_path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    base = _tune_paths(tmp_path, surface="same_block_closed")
    changed = _tune_paths(tmp_path, surface="same_block_closed", trial_count=40)

    assert base.study_id is not None
    assert changed.study_id == base.study_id


def test_study_and_artifact_ids_ignore_surface_name(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    clone_name = "same_block_closed_clone"
    _write_surface(conf_root, clone_name, _base_surface(conf_root))

    base_tune = _tune_paths(tmp_path, surface="same_block_closed")
    clone_tune = _tune_paths(tmp_path, surface=clone_name)
    base_train = _train_paths(tmp_path, surface="same_block_closed")
    clone_train = _train_paths(tmp_path, surface=clone_name)

    assert base_tune.study_id == clone_tune.study_id
    assert base_train.artifact_id == clone_train.artifact_id


def test_study_id_uses_objective_request_override(
    tmp_path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    base = _tune_paths(tmp_path, surface="same_block_closed")
    changed = _tune_paths(
        tmp_path,
        surface="same_block_closed",
        objective="validation_total_loss",
    )

    assert base.study_id is not None
    assert changed.study_id is not None
    assert changed.study_id != base.study_id


def test_artifact_id_uses_full_resolved_identity(
    tmp_path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    name = "artifact_identity_change"
    override = {"feature_set": "same_block_closed_no_time_since_start"}
    payload = _updated_surface(_base_surface(conf_root), override)
    _write_surface(conf_root, name, payload)

    base = _train_paths(tmp_path, surface="same_block_closed")
    changed = _train_paths(tmp_path, surface=name)

    assert base.artifact_id is not None
    assert changed.artifact_id is not None
    assert changed.artifact_id != base.artifact_id


def test_artifact_id_uses_objective_request_override(
    tmp_path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    base_train = _train_paths(tmp_path, surface="same_block_closed")
    changed_train = _train_paths(
        tmp_path,
        surface="same_block_closed",
        objective="validation_total_loss",
    )
    base_evaluate = _evaluate_paths(tmp_path, surface="same_block_closed")
    changed_evaluate = _evaluate_paths(
        tmp_path,
        surface="same_block_closed",
        objective="validation_total_loss",
    )

    assert base_train.artifact_id is not None
    assert changed_train.artifact_id is not None
    assert base_evaluate.artifact_id == base_train.artifact_id
    assert changed_evaluate.artifact_id == changed_train.artifact_id
    assert changed_train.artifact_id != base_train.artifact_id


def test_storage_root_does_not_affect_ids(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()
    first = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(surface="same_block_closed", storage_root=tmp_path / "one"),
        ),
    )
    second = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(surface="same_block_closed", storage_root=tmp_path / "two"),
        ),
    )

    first_paths = resolve_workflow_paths(first)
    second_paths = resolve_workflow_paths(second)

    assert first_paths.corpus_id == second_paths.corpus_id
    assert first_paths.artifact_id == second_paths.artifact_id
    assert first_paths.output_root != second_paths.output_root


def test_tuned_request_identity_matches_stored_study_manifest(
    tmp_path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()
    tune_config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowRequest(surface="same_block_closed", storage_root=tmp_path / "outputs"),
        ),
    )
    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(
                surface="same_block_closed",
                variant="tuned",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    tune_paths = resolve_workflow_paths(tune_config)
    manifest = manifest_from_tune_config(tune_config)

    assert tune_paths.study_id is not None
    assert study_request_identity_from_manifest(manifest) == (
        study_request_identity_from_tuned_config(
            train_config,
            study_id=tune_paths.study_id,
            dataset_id=tune_paths.corpus_id,
        )
    )


def test_corpus_id_uses_dataset_evaluation_date(tmp_path, isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    base = _train_paths(tmp_path, surface="same_block_closed")
    changed_dataset = dict(load_named_group("icdcs_2026", "dataset"))
    changed_dataset["evaluation_date"] = "2025-11-10"
    (conf_root / "dataset" / "icdcs_2026.yaml").write_text(
        yaml.safe_dump(changed_dataset, sort_keys=False),
        encoding="utf-8",
    )
    changed = _train_paths(tmp_path, surface="same_block_closed")

    assert changed.corpus_id != base.corpus_id


def test_tuned_training_spec_uses_resolved_workflow_paths(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()
    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(
                surface="same_block_closed",
                variant="tuned",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    paths = resolve_workflow_paths(config)
    tuned_config = cast(
        TrainConfig,
        apply_tuned_parameters(
            config,
            TunedParameterSet(
                training=TunedTrainingParams(learning_rate=0.001),
            ),
        ),
    )

    spec = build_training_spec(tuned_config, paths=paths)

    assert spec.study_id == paths.study_id
    assert spec.artifact_id == paths.artifact_id

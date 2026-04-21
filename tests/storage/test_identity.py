from __future__ import annotations

from typing import cast

import pytest
import yaml

from spice.config import (
    TrainConfig,
    TuneConfig,
    WorkflowRequest,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.registry import load_named_group
from spice.storage.identity import (
    study_request_identity_from_manifest,
    study_request_identity_from_tuned_config,
)
from spice.storage.layout import resolve_workflow_paths
from spice.storage.study_manifest import manifest_from_tune_config


def _write_preset(conf_root, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "preset" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_preset(conf_root) -> dict[str, object]:
    return load_named_group("icdcs_2026", "preset")


def _updated_preset(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    return {**base, **override}


def _train_paths(tmp_path, *, preset: str, variant: str | None = None):
    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(
                preset=preset,
                variant=variant,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )
    return resolve_workflow_paths(config)


def _tune_paths(tmp_path, *, preset: str):
    config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            WorkflowRequest(preset=preset, storage_root=tmp_path / "outputs"),
        ),
    )
    return resolve_workflow_paths(config)


@pytest.mark.parametrize(
    ("name", "override"),
    [
        (
            "objective_id",
            {"objective": "paper_profit_replay_2h", "evaluation": "paper_replay_2h"},
        ),
        (
            "training_id",
            {
                "training": {
                    "learning_rate": 0.0003,
                    "weight_decay": 0.01,
                    "batch_size": 256,
                    "max_epochs": 50,
                    "early_stopping": {"patience": 8, "min_delta": 0.0001},
                    "gradient_clip_norm": 1.0,
                    "seed": 2026,
                    "deterministic": True,
                    "log_every_n_steps": 100,
                    "input_normalization": {"id": "row_standard"},
                }
            },
        ),
        (
            "split_id",
            {"split": {"train_fraction": 0.7, "validation_fraction": 0.2}},
        ),
        (
            "tuning_id",
            {
                "tuning": {
                    "trial_count": 20,
                    "timeout_seconds": None,
                    "sampler_seed": 7,
                    "enable_pruning": True,
                }
            },
        ),
        ("tuning_space_id", {"tuning_space": "lstm_icdcs_2026_extensive"}),
    ],
)
def test_study_id_uses_full_resolved_identity(
    tmp_path,
    isolate_conf_root,
    name: str,
    override: dict[str, object],
) -> None:
    conf_root = isolate_conf_root()
    payload = _updated_preset(_base_preset(conf_root), override)
    _write_preset(conf_root, name, payload)

    base = _tune_paths(tmp_path, preset="icdcs_2026")
    changed = _tune_paths(tmp_path, preset=name)

    assert base.study_id is not None
    assert changed.study_id is not None
    assert changed.study_id != base.study_id


@pytest.mark.parametrize(
    ("name", "override", "variant"),
    [
        (
            "artifact_objective",
            {"objective": "paper_profit_replay_2h", "evaluation": "paper_replay_2h"},
            None,
        ),
        (
            "artifact_training",
            {
                "training": {
                    "learning_rate": 0.0003,
                    "weight_decay": 0.01,
                    "batch_size": 512,
                    "max_epochs": 50,
                    "early_stopping": {"patience": 8, "min_delta": 0.0001},
                    "gradient_clip_norm": 1.0,
                    "seed": 2026,
                    "deterministic": True,
                    "log_every_n_steps": 100,
                    "input_normalization": {"id": "window_weighted_standard"},
                }
            },
            None,
        ),
        ("artifact_model", {"model": "lstm"}, None),
        ("artifact_problem", {"problem": "icdcs_2026_paper_truth"}, None),
        ("artifact_feature", {"feature_set": "icdcs_2026_professor"}, None),
        ("artifact_prediction", {"prediction": "candidate_offset_selection"}, None),
        ("artifact_builder", {"dataset_builder": "professor_temporal"}, None),
        ("artifact_variant", {}, "tuned"),
    ],
)
def test_artifact_id_uses_full_resolved_identity(
    tmp_path,
    isolate_conf_root,
    name: str,
    override: dict[str, object],
    variant: str | None,
) -> None:
    conf_root = isolate_conf_root()
    payload = _updated_preset(_base_preset(conf_root), override)
    _write_preset(conf_root, name, payload)

    base = _train_paths(tmp_path, preset="icdcs_2026")
    changed = _train_paths(tmp_path, preset=name, variant=variant)

    assert base.artifact_id is not None
    assert changed.artifact_id is not None
    assert changed.artifact_id != base.artifact_id


def test_storage_root_does_not_affect_ids(tmp_path, isolate_conf_root) -> None:
    isolate_conf_root()
    first = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="icdcs_2026", storage_root=tmp_path / "one"),
        ),
    )
    second = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="icdcs_2026", storage_root=tmp_path / "two"),
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
            WorkflowRequest(preset="icdcs_2026", storage_root=tmp_path / "outputs"),
        ),
    )
    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(
                preset="icdcs_2026",
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

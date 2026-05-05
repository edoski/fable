from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from spice.config import (
    AcquireConfig,
    AcquireWorkflowSelection,
    EvaluateConfig,
    EvaluateWorkflowSelection,
    TrainConfig,
    TrainWorkflowSelection,
    TuneConfig,
    TuneWorkflowSelection,
    WorkflowTask,
    coerce_problem_spec,
    resolve_workflow_config,
)
from spice.config.groups import load_named_group_payload
from spice.core.errors import ConfigResolutionError

TEST_DATASET_ID = "cor_9a73b1e88edb488afb1e"
TEST_ARTIFACT_ID = "art_test"


def _write_surface(conf_root: Path, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "surface" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_surface(conf_root: Path) -> dict[str, object]:
    return load_named_group_payload("current_row_fee_dynamics", "surface")


def test_named_spec_identity_is_enforced_on_normal_load_paths(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    aliased_problem = conf_root / "problem" / "aliased_problem.yaml"
    aliased_problem.write_text(
        yaml.safe_dump(
            {
                "id": "different_problem",
                "lookback_seconds": 900,
                "sample_count": 1000000,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "observed_time_window",
                    "slot_spacing": {"id": "nominal"},
                },
                "execution_policy": {"id": "strict_deadline_miss"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigResolutionError,
        match="problem id must match spec name: aliased_problem",
    ):
        load_named_group_payload("aliased_problem", "problem")

    aliased_evaluation = conf_root / "evaluation" / "aliased_evaluation.yaml"
    aliased_evaluation.write_text(
        yaml.safe_dump(
            {
                "id": "poisson_replay_2h",
                "window_seconds": 7200,
                "arrival_rate_per_second": 0.01,
                "repetitions": 3,
                "seed": 2026,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigResolutionError,
        match="evaluation id must match spec name: aliased_evaluation",
    ):
        load_named_group_payload("aliased_evaluation", "evaluation")


def test_surface_refs_and_selection_defaults_resolve(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    child_root = tmp_path / "child_outputs"
    payload = _base_surface(conf_root)
    payload["objective"] = "profit_poisson_replay_2h"
    (conf_root / "training" / "child_training.yaml").write_text(
        yaml.safe_dump(
            {
                **load_named_group_payload("default", "training"),
                "batch_size": 64,
                "early_stopping": {
                    **cast(
                        dict[str, object],
                        load_named_group_payload("default", "training")["early_stopping"],
                    ),
                    "patience": 3,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (conf_root / "split" / "child_split.yaml").write_text(
        yaml.safe_dump(
            {
                **load_named_group_payload("default", "split"),
                "train_fraction": 0.7,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload["training"] = {"id": "child_training", "split": "child_split"}
    payload["storage"] = {"root": str(child_root)}
    payload["study"] = {"name": "child-study"}
    payload["artifact"] = {"variant": "baseline"}
    _write_surface(conf_root, "child_train", payload)

    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(surface="child_train", dataset_id=TEST_DATASET_ID),
        ),
    )

    assert config.training.batch_size == 64
    assert config.training.early_stopping.patience == 3
    assert config.training.early_stopping.min_delta == 0.0001
    assert config.split.train_fraction == 0.7
    assert config.storage.root == child_root
    assert config.study.name == "child-study"
    assert config.artifact.variant.value == "baseline"


def test_acquire_resolution_resolves_one_chain_specific_rpc_endpoint(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    config = cast(
        AcquireConfig,
        resolve_workflow_config(
            WorkflowTask.ACQUIRE,
            AcquireWorkflowSelection(
                surface="current_row_fee_dynamics",
                chain="avalanche",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.rpc_endpoint.provider_name == "publicnode"
    assert config.rpc_endpoint.url == "https://avalanche-c-chain-rpc.publicnode.com"
    assert config.rpc_endpoint.reference == "https://avalanche-c-chain-rpc.publicnode.com"


def test_acquire_resolution_fails_when_provider_lacks_chain_endpoint(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    (conf_root / "provider" / "eth_only.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "eth_only",
                "transport": {
                    "timeout_seconds": 30.0,
                    "retry_count": 5,
                    "backoff_factor": 0.125,
                },
                "acquisition": {
                    "dry_run": False,
                    "chunk_size": 4096,
                    "rpc": {
                        "batch_size": 16,
                        "concurrency": 1,
                        "min_batch_size": 1,
                        "concurrency_rungs": [1],
                    },
                },
                "endpoints": {
                    "ethereum": {
                        "url": "https://ethereum-rpc.publicnode.com",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = _base_surface(conf_root)
    payload["acquisition"] = {"provider": "eth_only"}
    _write_surface(conf_root, "eth_only_acquire", payload)

    with pytest.raises(
        ConfigResolutionError,
        match="provider eth_only does not define endpoint for avalanche",
    ):
        resolve_workflow_config(
            WorkflowTask.ACQUIRE,
            AcquireWorkflowSelection(
                surface="eth_only_acquire",
                chain="avalanche",
                storage_root=tmp_path / "outputs",
            ),
        )


def test_acquire_rejects_objective_override(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    with pytest.raises(ValidationError, match="objective"):
        AcquireWorkflowSelection.model_validate(
            {
                "surface": "current_row_fee_dynamics",
                "objective": "validation_total_loss",
                "storage_root": tmp_path / "outputs",
            }
        )


def test_test_workflow_loader_rejects_invalid_selection_override(
    tmp_path: Path,
    load_workflow_config,
) -> None:
    with pytest.raises(ValueError, match="not valid for acquire workflow"):
        load_workflow_config(
            WorkflowTask.ACQUIRE,
            workspace=tmp_path,
            override={"trial_count": 3},
        )


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        (None, "surface is required"),
        ("does_not_exist", "Unknown surface spec: does_not_exist"),
        ("broken", "Field required"),
    ],
)
def test_invalid_resolution_selections_fail_cleanly(
    tmp_path: Path,
    isolate_conf_root,
    surface: str | None,
    message: str,
) -> None:
    conf_root = isolate_conf_root()
    if surface == "broken":
        _write_surface(conf_root, "broken", {"chain": "ethereum"})

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface=surface,
                dataset_id=TEST_DATASET_ID,
                storage_root=tmp_path / "outputs",
            ),
        )


def test_benchmark_objective_requires_matching_evaluation(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    (conf_root / "objective" / "mismatch.yaml").write_text(
        "\n".join(
            [
                "id: evaluation",
                "metric_id: profit_over_baseline",
                "direction: maximize",
                "benchmark_id: other_evaluation",
            ]
        ),
        encoding="utf-8",
    )
    payload = _base_surface(conf_root)
    payload["objective"] = "mismatch"
    payload["evaluation"] = {"id": "poisson_replay_2h"}
    _write_surface(conf_root, "mismatch", payload)

    with pytest.raises(
        ConfigResolutionError,
        match=(
            "objective mismatch requires evaluation "
            "other_evaluation, got poisson_replay_2h"
        ),
    ):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="mismatch",
                dataset_id=TEST_DATASET_ID,
                storage_root=tmp_path / "outputs",
            ),
        )


def test_full_temporal_replay_objective_requires_matching_train_evaluation(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                objective="profit_full_temporal_replay",
                evaluation="full_temporal_replay",
                storage_root=tmp_path / "train_outputs",
            ),
        ),
    )
    assert train_config.objective.benchmark_id == "full_temporal_replay"
    assert train_config.evaluation is not None
    assert train_config.evaluation.id == "full_temporal_replay"

    with pytest.raises(
        ConfigResolutionError,
        match=(
            "objective profit_full_temporal_replay requires evaluation "
            "full_temporal_replay, got poisson_replay_2h"
        ),
    ):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                objective="profit_full_temporal_replay",
                evaluation="poisson_replay_2h",
                storage_root=tmp_path / "mismatch_outputs",
            ),
        )


def test_evaluate_resolves_eval_only_controls(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    config = cast(
        EvaluateConfig,
        resolve_workflow_config(
            WorkflowTask.EVALUATE,
            EvaluateWorkflowSelection(
                artifact_id=TEST_ARTIFACT_ID,
                dataset_id=TEST_DATASET_ID,
                evaluation="full_temporal_replay",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.artifact_id == TEST_ARTIFACT_ID
    assert config.dataset_id == TEST_DATASET_ID
    assert config.evaluation.id == "full_temporal_replay"


def test_tuned_train_rejects_dataset_id(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    with pytest.raises(ConfigResolutionError, match="tuned training must not define dataset_id"):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                study_id="stu_test",
                variant="tuned",
                storage_root=tmp_path / "outputs",
            ),
        )


def test_selection_overrides_allow_problem_features_and_evaluation_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                problem="current_row_recent_median",
                features="core_fee_dynamics",
                storage_root=tmp_path / "train_outputs",
            ),
        ),
    )
    assert train_config.problem.id == "current_row_recent_median"
    assert train_config.features.id == "core_fee_dynamics"
    assert train_config.evaluation is not None
    assert train_config.evaluation.id == "poisson_replay_2h"


def test_selection_accepts_inline_problem_spec(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()
    problem = coerce_problem_spec(
        {
            **load_named_group_payload("current_row_nominal", "problem"),
            "id": "inline_problem",
            "sample_count": 12345,
        }
    )

    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                problem=problem,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.problem.id == "inline_problem"
    assert config.problem.sample_count == 12345


def test_selection_overrides_allow_objective_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                objective="validation_total_loss",
                storage_root=tmp_path / "train_outputs",
            ),
        ),
    )
    assert train_config.objective.id == "validation"
    assert train_config.objective.metric_id == "total_loss"
    assert train_config.objective.direction == "minimize"

    tune_config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                objective="validation_total_loss",
                storage_root=tmp_path / "tune_outputs",
            ),
        ),
    )
    assert tune_config.objective.id == "validation"
    assert tune_config.objective.metric_id == "total_loss"


def test_selection_overrides_allow_model_and_tuning_space_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                model="transformer",
                tuning_space="transformer_default",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.model.id == "transformer"
    assert config.tuning_space.model.id == "transformer"
    assert type(config.model).__name__ == "TransformerModelConfig"
    assert type(config.tuning_space.model).__name__ == "TransformerTuningSpaceModelConfig"


@pytest.mark.parametrize(
    ("model", "tuning_space"),
    [
        ("lstm", "lstm_large_capacity"),
        ("transformer", "transformer_large_capacity"),
        ("transformer_lstm", "transformer_lstm_large_capacity"),
    ],
)
def test_large_capacity_tuning_spaces_resolve(
    model: str,
    tuning_space: str,
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowSelection(
                surface="current_row_fee_dynamics",
                dataset_id=TEST_DATASET_ID,
                model=model,
                tuning_space=tuning_space,
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.model.id == model
    assert config.tuning_space.model.id == model

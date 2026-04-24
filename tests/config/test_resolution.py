from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from spice.config import (
    AcquireConfig,
    AcquireWorkflowRequest,
    EvaluateConfig,
    EvaluateWorkflowRequest,
    TrainConfig,
    TrainWorkflowRequest,
    TuneConfig,
    TuneWorkflowRequest,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.registry import load_named_group
from spice.core.errors import ConfigResolutionError


def _write_surface(conf_root: Path, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "surface" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_surface(conf_root: Path) -> dict[str, object]:
    return load_named_group("same_block_closed", "surface")


def test_named_spec_identity_is_enforced_on_normal_load_paths(isolate_conf_root) -> None:
    conf_root = isolate_conf_root()
    aliased_problem = conf_root / "problem" / "aliased_problem.yaml"
    aliased_problem.write_text(
        yaml.safe_dump(
            {
                "id": "different_problem",
                "lookback_seconds": 900,
                "sample_count": 400000,
                "max_delay_seconds": 36,
                "compiler": {
                    "id": "estimated_block",
                    "lookback_interval_source": "nominal_chain_runtime",
                    "candidate_interval_source": "calibrated",
                    "calibrated_interval_statistic": "mean",
                },
                "realization_policy": {"id": "strict_deadline_miss"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigResolutionError,
        match="problem id must match spec name: aliased_problem",
    ):
        load_named_group("aliased_problem", "problem")


def test_surface_refs_and_request_defaults_resolve(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    child_root = tmp_path / "child_outputs"
    payload = _base_surface(conf_root)
    payload["objective"] = "profit_poisson_replay_2h_mean"
    payload["evaluation"] = "poisson_replay_2h_mean"
    (conf_root / "training" / "child_training.yaml").write_text(
        yaml.safe_dump(
            {
                **load_named_group("default", "training"),
                "batch_size": 64,
                "early_stopping": {
                    **cast(
                        dict[str, object],
                        load_named_group("default", "training")["early_stopping"],
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
                **load_named_group("default", "split"),
                "train_fraction": 0.7,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload["training"] = "child_training"
    payload["split"] = "child_split"
    payload["storage"] = {"root": str(child_root)}
    payload["study"] = {"name": "child-study"}
    payload["artifact"] = {"variant": "baseline"}
    _write_surface(conf_root, "child_train", payload)

    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(surface="child_train"),
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
            AcquireWorkflowRequest(
                surface="same_block_closed",
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
    payload["provider"] = "eth_only"
    _write_surface(conf_root, "eth_only_acquire", payload)

    with pytest.raises(
        ConfigResolutionError,
        match="provider eth_only does not define endpoint for avalanche",
    ):
        resolve_workflow_config(
            WorkflowTask.ACQUIRE,
            AcquireWorkflowRequest(
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
        AcquireWorkflowRequest(
            surface="same_block_closed",
            objective="validation_total_loss",
            storage_root=tmp_path / "outputs",
        )


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        (None, "surface is required"),
        ("does_not_exist", "Unknown surface spec: does_not_exist"),
        ("broken", "Field required"),
    ],
)
def test_invalid_resolution_requests_fail_cleanly(
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
            TrainWorkflowRequest(surface=surface, storage_root=tmp_path / "outputs"),
        )


def test_benchmark_objective_requires_matching_evaluation(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    payload = _base_surface(conf_root)
    payload["objective"] = "profit_poisson_replay_2h_mean"
    payload["evaluation"] = "fullset"
    _write_surface(conf_root, "mismatch", payload)

    with pytest.raises(
        ConfigResolutionError,
        match=(
            "objective profit_poisson_replay_2h_mean requires evaluation "
            "poisson_replay_2h_mean, got fullset"
        ),
    ):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(surface="mismatch", storage_root=tmp_path / "outputs"),
        )


def test_evaluate_allows_diagnostic_evaluation_to_differ_from_objective_benchmark(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    payload = _base_surface(conf_root)
    payload["objective"] = "profit_poisson_replay_2h_mean"
    payload["evaluation"] = "fullset"
    _write_surface(conf_root, "diagnostic", payload)

    config = resolve_workflow_config(
        WorkflowTask.EVALUATE,
        EvaluateWorkflowRequest(surface="diagnostic", storage_root=tmp_path / "outputs"),
    )

    assert config.objective.id == "evaluation"
    assert config.objective.benchmark_id == "poisson_replay_2h_mean"
    assert config.evaluation.id == "fullset"


def test_request_overrides_allow_problem_feature_set_and_evaluation_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(
                surface="block_open_lagged",
                problem="current_row_recent_delta_window",
                feature_set="block_open_lagged_no_time_since_start",
                storage_root=tmp_path / "train_outputs",
            ),
        ),
    )
    assert train_config.problem.id == "current_row_recent_delta_window"
    assert train_config.feature_set.id == "block_open_lagged_no_time_since_start"
    assert train_config.evaluation is not None
    assert train_config.evaluation.id == "poisson_replay_2h_mean"

    evaluate_config = resolve_workflow_config(
        WorkflowTask.EVALUATE,
        EvaluateWorkflowRequest(
            surface="same_block_closed",
            evaluation="anchor_basefee_fullset",
            storage_root=tmp_path / "eval_outputs",
        ),
    )
    assert evaluate_config.evaluation is not None
    assert evaluate_config.evaluation.id == "anchor_basefee_fullset"


def test_request_overrides_allow_objective_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            TrainWorkflowRequest(
                surface="same_block_closed",
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
            TuneWorkflowRequest(
                surface="same_block_closed",
                objective="validation_total_loss",
                storage_root=tmp_path / "tune_outputs",
            ),
        ),
    )
    assert tune_config.objective.id == "validation"
    assert tune_config.objective.metric_id == "total_loss"

    evaluate_config = cast(
        EvaluateConfig,
        resolve_workflow_config(
            WorkflowTask.EVALUATE,
            EvaluateWorkflowRequest(
                surface="same_block_closed",
                objective="validation_total_loss",
                storage_root=tmp_path / "eval_outputs",
            ),
        ),
    )
    assert evaluate_config.objective.id == "validation"
    assert evaluate_config.evaluation is not None
    assert evaluate_config.evaluation.id == "poisson_replay_2h_mean"


def test_request_overrides_allow_model_and_tuning_space_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    config = cast(
        TuneConfig,
        resolve_workflow_config(
            WorkflowTask.TUNE,
            TuneWorkflowRequest(
                surface="same_block_closed",
                model="transformer",
                tuning_space="transformer_default",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.model.id == "transformer"
    assert config.tuning_space.model.id == "transformer"

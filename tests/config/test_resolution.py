from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from spice.config import (
    AcquireConfig,
    TrainConfig,
    WorkflowRequest,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.registry import load_named_group
from spice.core.errors import ConfigResolutionError


def _write_preset(conf_root: Path, name: str, payload: dict[str, object]) -> None:
    path = conf_root / "preset" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_preset(conf_root: Path) -> dict[str, object]:
    return load_named_group("icdcs_2026", "preset")


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


def test_preset_frames_are_explicit_and_request_overrides_are_narrow(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    child_root = tmp_path / "child_outputs"
    payload = _base_preset(conf_root)
    payload["objective"] = "paper_profit_replay_2h"
    payload["evaluation"] = "paper_replay_2h"
    payload["training"] = {
        **cast(dict[str, object], payload["training"]),
        "batch_size": 64,
        "early_stopping": {
            **cast(
                dict[str, object],
                cast(dict[str, object], payload["training"])["early_stopping"],
            ),
            "patience": 3,
        },
    }
    payload["split"] = {
        **cast(dict[str, object], payload["split"]),
        "train_fraction": 0.7,
    }
    payload["storage"] = {"root": str(child_root)}
    payload["study"] = {"name": "child-study"}
    payload["artifact"] = {"variant": "baseline"}
    _write_preset(conf_root, "child_train", payload)

    config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="child_train"),
        ),
    )

    assert config.objective.metric_id == "profit_over_baseline"
    assert config.objective.direction == "maximize"
    assert config.training.learning_rate == 0.0003
    assert config.training.batch_size == 64
    assert config.training.early_stopping.patience == 3
    assert config.training.early_stopping.min_delta == 0.0001
    assert config.split.train_fraction == 0.7
    assert config.split.validation_fraction == 0.1
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
            WorkflowRequest(
                preset="icdcs_2026",
                chain="avalanche",
                storage_root=tmp_path / "outputs",
            ),
        ),
    )

    assert config.rpc_endpoint.provider_name == "publicnode"
    assert config.rpc_endpoint.url == "https://avalanche-c-chain-rpc.publicnode.com"
    assert config.rpc_endpoint.reference == "https://avalanche-c-chain-rpc.publicnode.com"
    assert config.rpc_endpoint.timeout_seconds == 30.0
    assert config.rpc_endpoint.retry_count == 5
    assert config.rpc_endpoint.backoff_factor == 0.125


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
    payload = _base_preset(conf_root)
    payload["provider"] = "eth_only"
    _write_preset(conf_root, "eth_only_acquire", payload)

    with pytest.raises(
        ConfigResolutionError,
        match="provider eth_only does not define endpoint for avalanche",
    ):
        resolve_workflow_config(
            WorkflowTask.ACQUIRE,
            WorkflowRequest(
                preset="eth_only_acquire",
                chain="avalanche",
                storage_root=tmp_path / "outputs",
            ),
        )


def test_resolution_requires_preset(isolate_conf_root) -> None:
    isolate_conf_root()

    with pytest.raises(ConfigResolutionError, match="preset is required"):
        resolve_workflow_config(WorkflowTask.TRAIN, WorkflowRequest())


def test_unknown_preset_reports_clean_error(tmp_path: Path, isolate_conf_root) -> None:
    isolate_conf_root()

    with pytest.raises(ConfigResolutionError, match="Unknown preset spec: does_not_exist"):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="does_not_exist", storage_root=tmp_path / "outputs"),
        )


def test_preset_validation_reports_missing_required_fields(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    _write_preset(conf_root, "broken", {"chain": "ethereum"})

    with pytest.raises(ConfigResolutionError, match="Field required"):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="broken", storage_root=tmp_path / "outputs"),
        )


def test_benchmark_objective_requires_matching_evaluation(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    payload = _base_preset(conf_root)
    payload["objective"] = "paper_profit_replay_2h"
    payload["evaluation"] = "paper_fullset"
    _write_preset(conf_root, "mismatch", payload)

    with pytest.raises(
        ConfigResolutionError,
        match=(
            "objective paper_profit_replay_2h requires evaluation "
            "paper_replay_2h, got paper_fullset"
        ),
    ):
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(preset="mismatch", storage_root=tmp_path / "outputs"),
        )


def test_evaluate_allows_diagnostic_evaluation_to_differ_from_objective_benchmark(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    conf_root = isolate_conf_root()
    payload = _base_preset(conf_root)
    payload["objective"] = "paper_profit_replay_2h"
    payload["evaluation"] = "paper_fullset"
    _write_preset(conf_root, "diagnostic", payload)

    config = resolve_workflow_config(
        WorkflowTask.EVALUATE,
        WorkflowRequest(preset="diagnostic", storage_root=tmp_path / "outputs"),
    )

    assert config.objective.id == "evaluation"
    assert config.objective.benchmark_id == "paper_replay_2h"
    assert config.evaluation.id == "paper_fullset"


def test_request_overrides_allow_problem_feature_set_and_evaluation_selection(
    tmp_path: Path,
    isolate_conf_root,
) -> None:
    isolate_conf_root()

    train_config = cast(
        TrainConfig,
        resolve_workflow_config(
            WorkflowTask.TRAIN,
            WorkflowRequest(
                preset="icdcs_2026_oracle_block_open",
                problem="icdcs_2026_oracle_intermediate_recent_deltas",
                feature_set="icdcs_2026_professor_block_open_no_time_since_start",
                storage_root=tmp_path / "train_outputs",
            ),
        ),
    )
    assert train_config.problem.id == "icdcs_2026_oracle_intermediate_recent_deltas"
    assert train_config.feature_set.id == "icdcs_2026_professor_block_open_no_time_since_start"
    assert train_config.evaluation is not None
    assert train_config.evaluation.id == "paper_replay_2h"

    evaluate_config = resolve_workflow_config(
        WorkflowTask.EVALUATE,
        WorkflowRequest(
            preset="icdcs_2026_oracle_intermediate",
            evaluation="notebook_basefee_fullset",
            storage_root=tmp_path / "eval_outputs",
        ),
    )
    assert evaluate_config.evaluation is not None
    assert evaluate_config.evaluation.id == "notebook_basefee_fullset"

from __future__ import annotations

import pytest

from spice.core.config import WorkflowTask
from spice.workflows.dvc import load_stage_config
from tests.support import (
    REPO_ROOT,
    TEST_WINDOW_END_TIMESTAMP,
    TEST_WINDOW_START_TIMESTAMP,
    base_overrides,
    compose_experiment,
)


def test_hydra_train_config_composes_and_resolves_paths(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path)
        + ["dataset.temporal.max_delay_seconds=24", "model=transformer"],
    )

    assert config.task is WorkflowTask.TRAIN
    assert config.dataset.temporal.max_delay_seconds == 24
    assert config.model.family.value == "transformer"
    assert config.dataset.id == "icdcs_2025_11_09"
    assert config.dataset.sampling.history_anchor_count is None
    assert config.dataset.sampling.effective_history_anchor_count == 48
    assert config.paths.artifact_root.as_posix().endswith(
        "/ethereum/icdcs_2025_11_09/transformer/24s"
    )
    assert config.paths.history_dir.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/history"
    )
    assert config.paths.metadata_root.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/.spice"
    )
    assert config.paths.dataset_metadata_path.as_posix().endswith(
        "/datasets/ethereum/icdcs_2025_11_09/.spice/metadata.json"
    )


def test_direct_provider_reads_env_interpolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ETHEREUM_RPC_URL", "https://eth.example.test")
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=direct"],
    )

    assert config.provider.endpoint_for(config.chain.name) == "https://eth.example.test"
    assert config.provider.reference_for(config.chain.name) == "$ETHEREUM_RPC_URL"
    assert config.acquisition.chunk_size == 500
    assert config.acquisition.rpc_batch_size == 256


def test_train_config_does_not_require_direct_provider_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["provider=direct"],
    )

    assert config.provider.reference_for(config.chain.name) == "$ETHEREUM_RPC_URL"


def test_acquire_config_requires_direct_provider_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)

    with pytest.raises(ValueError, match="Missing RPC endpoint for chain: ethereum"):
        compose_experiment(
            "acquire",
            overrides=base_overrides(tmp_path) + ["provider=direct"],
        )


@pytest.mark.parametrize(
    ("chain_name", "uses_poa_extra_data", "rpc_batch_size", "chunk_size"),
    [
        ("ethereum", False, 128, 500),
        ("polygon", True, 32, 500),
        ("avalanche", True, 128, 500),
    ],
)
def test_publicnode_chain_matrix_applies_expected_chain_settings(
    tmp_path,
    chain_name: str,
    uses_poa_extra_data: bool,
    rpc_batch_size: int,
    chunk_size: int,
) -> None:
    overrides = [
        f"runtime.output_root={tmp_path / 'artifacts'}",
        "tracking.enabled=false",
    ]
    if chain_name != "ethereum":
        overrides.append(f"chain={chain_name}")

    config = compose_experiment("acquire", overrides=overrides)

    assert config.chain.name.value == chain_name
    assert config.chain.uses_poa_extra_data is uses_poa_extra_data
    assert config.acquisition.rpc_batch_size == rpc_batch_size
    assert config.acquisition.chunk_size == chunk_size


def test_invalid_transformer_config_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by nhead"):
        compose_experiment(
            "train",
            overrides=base_overrides(tmp_path)
            + ["model=transformer", "model.d_model=126", "model.nhead=8"],
        )


def test_date_window_resolves_to_half_open_utc_timestamps(tmp_path) -> None:
    config = compose_experiment("simulate", overrides=base_overrides(tmp_path))

    assert config.dataset.window.start_timestamp == TEST_WINDOW_START_TIMESTAMP
    assert config.dataset.window.end_timestamp == TEST_WINDOW_END_TIMESTAMP


def test_history_anchor_count_cannot_be_smaller_than_anchor_count(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="history_anchor_count must be at least dataset.sampling.anchor_count",
    ):
        compose_experiment(
            "acquire",
            overrides=base_overrides(tmp_path)
            + ["dataset.sampling.history_anchor_count=32"],
        )


def test_invalid_tuning_direction_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="maximize|minimize"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path) + ["tuning.direction=sideways"],
        )


def test_invalid_tuning_objective_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_loss|validation_accuracy"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path) + ["tuning.objective_metric=validation_magic"],
        )


def test_tuning_search_space_rejects_empty_candidate_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path)
            + ["tuning.search_space.training.learning_rate=[]"],
        )


def test_tuning_search_space_rejects_unsupported_field(tmp_path) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        compose_experiment(
            "tune",
            overrides=base_overrides(tmp_path)
            + ["+tuning.search_space.training.momentum=[0.9]"],
        )


def test_dvc_runner_loads_generated_params_and_forces_stage_task() -> None:
    config = load_stage_config("train", REPO_ROOT / "params.yaml")

    assert config.task is WorkflowTask.TRAIN
    assert config.tuning.apply_best_params is True
    assert config.paths.artifact_root.as_posix().endswith(
        "/models/ethereum/icdcs_2025_11_09/lstm/36s"
    )

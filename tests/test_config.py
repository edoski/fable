from __future__ import annotations

import pytest

from spice.core.constants import DEFAULT_WINDOW_END_TIMESTAMP, DEFAULT_WINDOW_START_TIMESTAMP
from spice.workflows.dvc import load_stage_config
from tests.support import REPO_ROOT, base_overrides, compose_experiment


def test_hydra_train_config_composes_and_resolves_paths(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path)
        + ["dataset.temporal.max_delay_seconds=24", "model=transformer"],
    )

    assert config.task == "train"
    assert config.dataset.temporal.max_delay_seconds == 24
    assert config.model.family.value == "transformer"
    assert config.dataset.id == "icdcs_2025_11_09"
    assert config.dataset.sampling.history_anchor_count is None
    assert config.dataset.sampling.effective_history_anchor_count == 48
    assert config.paths.artifact_root.endswith("/ethereum/icdcs_2025_11_09/transformer/24s")
    assert config.paths.raw_history_dir.endswith(
        "/datasets/ethereum/icdcs_2025_11_09/raw/history"
    )
    assert config.paths.metadata_root.endswith("/datasets/ethereum/icdcs_2025_11_09/.spice")
    assert config.paths.dataset_metadata_path.endswith(
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


def test_avalanche_config_enables_poa_extra_data_middleware(tmp_path) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["chain=avalanche"],
    )

    assert config.chain.uses_poa_extra_data


def test_invalid_transformer_config_fails_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by nhead"):
        compose_experiment(
            "train",
            overrides=base_overrides(tmp_path)
            + ["model=transformer", "model.d_model=126", "model.nhead=8"],
        )


def test_date_window_resolves_to_half_open_utc_timestamps(tmp_path) -> None:
    config = compose_experiment("simulate", overrides=base_overrides(tmp_path))

    assert config.dataset.window.start_timestamp == DEFAULT_WINDOW_START_TIMESTAMP
    assert config.dataset.window.end_timestamp == DEFAULT_WINDOW_END_TIMESTAMP


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


def test_dvc_runner_loads_generated_params_and_forces_stage_task() -> None:
    config = load_stage_config("train", REPO_ROOT / "params.yaml")

    assert config.task == "train"
    assert config.tuning.apply_best_params is True
    assert config.paths.artifact_root.endswith("/models/ethereum/icdcs_2025_11_09/lstm/36s")

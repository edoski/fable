from __future__ import annotations

import pytest

from tests.support import base_overrides, compose_experiment


def test_hydra_train_config_composes_and_resolves_paths(tmp_path) -> None:
    config = compose_experiment(
        "train",
        overrides=base_overrides(tmp_path) + ["max_delay_seconds=24", "model=transformer"],
    )

    assert config.task == "train"
    assert config.max_delay_seconds == 24
    assert config.model.family.value == "transformer"
    assert config.paths.artifact_root.endswith("/transformer/24s")
    assert config.paths.raw_history_dir.endswith("/datasets/ethereum/raw/history")
    assert config.paths.metadata_root.endswith("/datasets/ethereum/.spice")
    assert config.paths.dataset_metadata_path.endswith("/datasets/ethereum/.spice/metadata.json")


def test_direct_provider_reads_env_interpolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ETHEREUM_RPC_URL", "https://eth.example.test")
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=direct"],
    )

    assert config.provider.endpoint_for(config.chain.name) == "https://eth.example.test"
    assert config.provider.reference_for(config.chain.name) == "$ETHEREUM_RPC_URL"


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

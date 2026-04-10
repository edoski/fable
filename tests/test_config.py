from __future__ import annotations

import pytest
import yaml

from spice.core.config import ChainName, ExperimentConfig
from tests.support import build_test_config, write_config


def test_experiment_config_loads_and_resolves_chain(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "artifacts"
    write_config(config_path, output_root=output_root)

    config = ExperimentConfig.load(config_path)

    assert config.output_root == output_root
    assert config.lookback_seconds == 120
    assert config.max_delay_seconds == [36]
    assert config.resolve_chain(ChainName.ETHEREUM).chain_id == 1


def test_experiment_config_rejects_invalid_split_sum(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = build_test_config(tmp_path / "artifacts")
    payload["split"] = {"train_fraction": 0.8, "validation_fraction": 0.2}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be less than 1"):
        ExperimentConfig.load(config_path)


def test_experiment_config_rejects_unknown_top_level_keys(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = build_test_config(tmp_path / "artifacts")
    payload["unexpected_option"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        ExperimentConfig.load(config_path)

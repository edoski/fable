import tempfile
import unittest
from pathlib import Path

import yaml

from spice_temporal.config import ExperimentConfig
from tests.support import build_test_config


class ConfigTestCase(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/baseline.yaml"))
        self.assertEqual(config.lookback_seconds, 600)
        self.assertEqual(config.max_delay_seconds, [12, 24, 36])
        self.assertEqual(len(config.chains), 3)
        self.assertEqual(config.chains[2].block_time_seconds, 1.6)
        self.assertEqual(config.chains[2].history_days, 10)
        self.assertEqual(config.target_anchor_count, 400_000)
        self.assertEqual(config.pull.requests_per_second, 25)
        self.assertEqual(config.pull.max_concurrent_requests, 2)
        self.assertEqual(config.pull.max_concurrent_chunks, 1)
        self.assertEqual(config.simulation.window_seconds, 7_200)
        self.assertEqual(config.simulation.arrival_rate_per_second, 0.05)
        self.assertEqual(config.simulation.repetitions, 50)

    def test_pilot_config_loads(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/pilots/ethereum-36s.yaml"))
        self.assertEqual(config.output_root, Path("./artifacts/pilots/ethereum-36s"))
        self.assertEqual(config.max_delay_seconds, [36])
        self.assertEqual(config.target_anchor_count, 5_000)
        self.assertEqual(len(config.chains), 1)
        self.assertEqual(config.chains[0].name, "ethereum")
        self.assertEqual(config.chains[0].block_time_seconds, 12.0)
        self.assertEqual(config.chains[0].history_days, 1)

    def test_config_rejects_invalid_split_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            config = build_test_config()
            config["split"] = {"train_fraction": 0.8, "validation_fraction": 0.2}
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be less than 1"):
                ExperimentConfig.from_yaml(path)

    def test_config_rejects_unknown_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            config = build_test_config()
            config["extra"] = True
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level keys"):
                ExperimentConfig.from_yaml(path)


if __name__ == "__main__":
    unittest.main()

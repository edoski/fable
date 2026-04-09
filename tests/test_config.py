import unittest
from pathlib import Path

from spice_temporal.config import ExperimentConfig


class ConfigTestCase(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/baseline.yaml"))
        self.assertEqual(config.lookback_seconds, 600)
        self.assertEqual(config.window_seconds, [12, 24, 36])
        self.assertEqual(len(config.chains), 3)
        self.assertEqual(config.pull.requests_per_second, 10)
        self.assertEqual(config.pull.max_concurrent_requests, 2)
        self.assertEqual(config.pull.max_concurrent_chunks, 1)

    def test_pilot_config_loads(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/pilots/ethereum-36s.yaml"))
        self.assertEqual(config.output_root, Path("./artifacts/pilots/ethereum-36s"))
        self.assertEqual(config.window_seconds, [36])
        self.assertEqual(len(config.chains), 1)
        self.assertEqual(config.chains[0].name, "ethereum")
        self.assertEqual(config.chains[0].history_days_hint, 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from spice_temporal.config import ExperimentConfig
from spice_temporal.cryo import build_pull_plan


class CryoPlanTestCase(unittest.TestCase):
    def test_pull_plan_includes_rate_controls(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/pilots/ethereum-36s.yaml"))
        plans = build_pull_plan(config)
        self.assertEqual(len(plans), 1)
        command = plans[0].command
        self.assertIn("--requests-per-second 10", command)
        self.assertIn("--max-concurrent-requests 2", command)
        self.assertIn("--max-concurrent-chunks 1", command)


if __name__ == "__main__":
    unittest.main()

import os
import shlex
import unittest
from pathlib import Path
from unittest.mock import patch

from spice_temporal.config import ChainConfig, ExperimentConfig, PullConfig
from spice_temporal.cryo import (
    build_cryo_args,
    build_cryo_command,
    build_pull_plan,
    evaluation_range,
)


class CryoPlanTestCase(unittest.TestCase):
    def test_pull_plan_includes_rate_controls(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/pilots/ethereum-36s.yaml"))
        plans = build_pull_plan(config)
        self.assertEqual(len(plans), 1)
        command = plans[0].command
        self.assertIn("--requests-per-second 10", command)
        self.assertIn("--max-concurrent-requests 2", command)
        self.assertIn("--max-concurrent-chunks 1", command)

    def test_command_string_and_args_share_one_command_spec(self) -> None:
        chain = ChainConfig(
            name="ethereum",
            chain_id=1,
            block_time_seconds=12.0,
            history_days=1,
        )
        pull = PullConfig(
            requests_per_second=10,
            max_concurrent_requests=2,
            max_concurrent_chunks=1,
        )
        output_dir = Path("artifacts/raw/ethereum/history")
        with patch.dict(os.environ, {"ALCHEMY_API_KEY": "test-key"}):
            args = build_cryo_args(chain, pull, output_dir, evaluation_range(), overwrite=True)
            command = build_cryo_command(
                chain,
                pull,
                output_dir,
                evaluation_range(),
                overwrite=True,
            )

        self.assertEqual(
            shlex.split(command.replace("$ALCHEMY_API_KEY", "test-key")),
            args,
        )


if __name__ == "__main__":
    unittest.main()

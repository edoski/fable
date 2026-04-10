import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from spice_temporal.api import (
    load_artifact,
    load_config,
    run_simulation_workflow,
    run_training_workflow,
)
from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from tests.support import make_evaluation_block, make_history_block, write_config


class ApiWorkflowTestCase(unittest.TestCase):
    def test_high_level_api_runs_training_and_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            history_dir = tmp_path / "history"
            history_dir.mkdir()
            (history_dir / "blocks.json").write_text(
                json.dumps([asdict(make_history_block(index)) for index in range(420)]),
                encoding="utf-8",
            )

            evaluation_dir = tmp_path / "evaluation"
            evaluation_dir.mkdir()
            (evaluation_dir / "blocks.json").write_text(
                json.dumps([asdict(make_evaluation_block(index)) for index in range(720)]),
                encoding="utf-8",
            )

            config = load_config(config_path)
            artifact_dir = tmp_path / "artifact"
            train_report = run_training_workflow(
                config,
                history_dir,
                artifact_dir,
                "ethereum",
                "lstm",
                36,
                device="cpu",
            )
            self.assertEqual(train_report.chain, "ethereum")
            self.assertEqual(train_report.family, "lstm")
            self.assertTrue((artifact_dir / TRAIN_REPORT_FILENAME).exists())

            loaded_artifact = load_artifact(artifact_dir)
            self.assertEqual(loaded_artifact.manifest.max_delay_seconds, 36)

            simulation_report = run_simulation_workflow(
                config,
                artifact_dir,
                history_dir,
                evaluation_dir,
                device="cpu",
            )
            self.assertEqual(simulation_report.chain, "ethereum")
            self.assertGreater(simulation_report.total_events, 0)
            self.assertTrue((artifact_dir / SIMULATION_REPORT_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()

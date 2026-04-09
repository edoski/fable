import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from spice_temporal.cli import app
from spice_temporal.constants import EVALUATION_START_TS
from spice_temporal.io import write_rows
from spice_temporal.raw_validation import RawPullValidationReport
from spice_temporal.records import BlockRecord


def make_history_block(index: int) -> BlockRecord:
    return BlockRecord(
        block_number=index,
        timestamp=EVALUATION_START_TS - 12 * (420 - index),
        base_fee_per_gas=100 + ((index // 3) % 7),
        gas_used=15_000_000 + (index % 1000),
        gas_limit=30_000_000,
        chain_id=1,
    )


def make_evaluation_block(index: int) -> BlockRecord:
    return BlockRecord(
        block_number=1_000 + index,
        timestamp=EVALUATION_START_TS + 12 * index,
        base_fee_per_gas=120 + ((index // 5) % 9),
        gas_used=15_100_000 + (index % 1000),
        gas_limit=30_000_000,
        chain_id=1,
    )


def write_config(path: Path, *, output_root: Path | None = None) -> None:
    config = {
        "output_root": str(output_root or Path("./artifacts/test")),
        "max_delay_seconds": [36],
        "lookback_seconds": 600,
        "target_anchor_count": 64,
        "pull": {
            "requests_per_second": 10,
            "max_concurrent_requests": 2,
            "max_concurrent_chunks": 1,
        },
        "split": {
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "training": {
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "effective_batch_size": 8,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0001,
            "gradient_clip_norm": 1.0,
            "alpha": 1.0,
            "beta": 0.25,
            "device": "cpu",
            "seed": 2026,
        },
        "simulation": {
            "window_seconds": 600,
            "arrival_rate_per_second": 0.02,
            "repetitions": 3,
            "seed": 2026,
        },
        "chains": [
            {
                "name": "ethereum",
                "chain_id": 1,
                "block_time_seconds": 12.0,
                "history_days": 1,
            }
        ],
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


class CliTrainingTestCase(unittest.TestCase):
    def test_pull_blocks_rejects_unknown_segment_choice(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts")
            result = runner.invoke(
                app,
                ["pull-blocks", str(config_path), "ethereum", "invalid-segment"],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for", result.output)

    def test_train_rejects_unknown_model_family_choice(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            result = runner.invoke(
                app,
                [
                    "train",
                    str(config_path),
                    str(tmp_path / "history"),
                    str(tmp_path / "artifact"),
                    "ethereum",
                    "invalid-family",
                    "36",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for", result.output)

    def test_train_and_simulate_write_artifact_reports(self) -> None:
        runner = CliRunner()
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

            artifact_dir = tmp_path / "artifact"
            train_result = runner.invoke(
                app,
                [
                    "train",
                    str(config_path),
                    str(history_dir),
                    str(artifact_dir),
                    "ethereum",
                    "lstm",
                    "36",
                    "--device",
                    "cpu",
                ],
            )
            self.assertEqual(train_result.exit_code, 0, msg=train_result.stdout)

            train_report = json.loads(
                (artifact_dir / TRAIN_REPORT_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(train_report["chain"], "ethereum")
            self.assertEqual(train_report["family"], "lstm")
            self.assertEqual(train_report["target_anchor_count"], 64)
            self.assertEqual(train_report["n_examples_total"], 64)
            self.assertEqual(train_report["action_count"], 4)

            simulate_result = runner.invoke(
                app,
                [
                    "simulate",
                    str(config_path),
                    str(artifact_dir),
                    str(history_dir),
                    str(evaluation_dir),
                    "--device",
                    "cpu",
                ],
            )
            self.assertEqual(simulate_result.exit_code, 0, msg=simulate_result.stdout)
            simulation_report = json.loads(
                (artifact_dir / SIMULATION_REPORT_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(simulation_report["chain"], "ethereum")
        self.assertEqual(simulation_report["family"], "lstm")
        self.assertEqual(simulation_report["max_delay_seconds"], 36)
        self.assertEqual(simulation_report["repetitions"], 3)
        self.assertEqual(simulation_report["action_count"], 4)
        self.assertGreater(simulation_report["n_examples_total"], 0)
        self.assertGreater(simulation_report["total_events"], 0)

    def test_validate_pull_succeeds_on_clean_raw_dataset(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_root = tmp_path / "artifacts"
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=output_root)

            raw_dir = output_root / "raw" / "ethereum" / "history"
            raw_dir.mkdir(parents=True)
            history_start = EVALUATION_START_TS - 24 * 60 * 60
            write_rows(
                raw_dir / "ethereum__blocks__1_to_3.parquet",
                [
                    {
                        "block_number": 1,
                        "timestamp": history_start + 1,
                        "base_fee_per_gas": 100,
                        "gas_used": 15_000_001,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 2,
                        "timestamp": history_start + 13,
                        "base_fee_per_gas": 101,
                        "gas_used": 15_000_002,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 3,
                        "timestamp": history_start + 25,
                        "base_fee_per_gas": 102,
                        "gas_used": 15_000_003,
                        "chain_id": 1,
                    },
                ],
            )

            result = runner.invoke(app, ["validate-pull", str(config_path), "ethereum", "history"])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("status=clean", result.stdout)

    def test_pull_blocks_validate_on_success_invokes_validation_once(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts")
            report = RawPullValidationReport(
                dataset_path=tmp_path / "artifacts" / "raw" / "ethereum" / "history",
                expected_start_timestamp=1,
                expected_end_timestamp=2,
            )

            with (
                patch("spice_temporal.cli.run_cryo") as run_cryo_mock,
                patch("spice_temporal.cli.validate_raw_pull", return_value=report) as validate_mock,
            ):
                run_cryo_mock.return_value = subprocess.CompletedProcess(
                    args=["cryo"],
                    returncode=0,
                    stdout="",
                    stderr="",
                )
                result = runner.invoke(
                    app,
                    [
                        "pull-blocks",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--no-dry-run",
                        "--validate-on-success",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        validate_mock.assert_called_once()

    def test_pull_blocks_validate_on_success_returns_non_zero_on_validation_error(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts")
            report = RawPullValidationReport(
                dataset_path=tmp_path / "artifacts" / "raw" / "ethereum" / "history",
                expected_start_timestamp=1,
                expected_end_timestamp=2,
                status="error",
                errors=["gap"],
            )

            with (
                patch("spice_temporal.cli.run_cryo") as run_cryo_mock,
                patch("spice_temporal.cli.validate_raw_pull", return_value=report),
            ):
                run_cryo_mock.return_value = subprocess.CompletedProcess(
                    args=["cryo"],
                    returncode=0,
                    stdout="",
                    stderr="",
                )
                result = runner.invoke(
                    app,
                    [
                        "pull-blocks",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--no-dry-run",
                        "--validate-on-success",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("status=error", result.stdout)


if __name__ == "__main__":
    unittest.main()

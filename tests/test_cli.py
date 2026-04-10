import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from spice_temporal.api import load_config
from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from spice_temporal.cli import app
from spice_temporal.config import BlockSegment, PullConfig
from spice_temporal.constants import EVALUATION_START_TS
from spice_temporal.cryo import history_range_for_chain
from spice_temporal.io import write_rows
from spice_temporal.provenance import source_manifest_path_for, write_source_manifest
from spice_temporal.raw_validation import RawPullValidationReport
from spice_temporal.rpc_providers import RpcProvider, RpcProviderName
from tests.support import make_evaluation_block, make_history_block, write_config


class CliTrainingTestCase(unittest.TestCase):
    def test_pull_blocks_rejects_unknown_segment_choice(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts")
            result = runner.invoke(
                app,
                ["blocks", "pull", str(config_path), "ethereum", "invalid-segment"],
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

            result = runner.invoke(
                app,
                ["blocks", "validate", str(config_path), "ethereum", "history"],
            )
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

            with patch(
                "spice_temporal.cli._pull_blocks",
                return_value=SimpleNamespace(
                    process=SimpleNamespace(stdout="", stderr=""),
                    validation=report,
                    source_manifest_path=None,
                ),
            ) as pull_blocks_mock:
                result = runner.invoke(
                    app,
                    [
                        "blocks",
                        "pull",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--no-dry-run",
                        "--validate-on-success",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        pull_blocks_mock.assert_called_once()

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

            with patch(
                "spice_temporal.cli._pull_blocks",
                return_value=SimpleNamespace(
                    process=SimpleNamespace(stdout="", stderr=""),
                    validation=report,
                    source_manifest_path=None,
                ),
            ):
                result = runner.invoke(
                    app,
                    [
                        "blocks",
                        "pull",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--no-dry-run",
                        "--validate-on-success",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("status=error", result.stdout)

    def test_stage_blocks_invokes_stage_pull_once(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts" / "baseline")
            staged_dir = (
                tmp_path
                / "artifacts"
                / "staging"
                / "publicnode"
                / "raw"
                / "ethereum"
                / "history"
            )
            report = RawPullValidationReport(
                dataset_path=staged_dir,
                expected_start_timestamp=1,
                expected_end_timestamp=2,
            )

            with patch(
                "spice_temporal.cli._stage_block_pull",
                return_value=SimpleNamespace(
                    process=SimpleNamespace(stdout="", stderr=""),
                    validation=report,
                    source_manifest_path=source_manifest_path_for(staged_dir),
                ),
            ) as stage_blocks_mock:
                result = runner.invoke(
                    app,
                    [
                        "blocks",
                        "stage",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--rpc-provider",
                        "publicnode",
                        "--no-dry-run",
                        "--validate-on-success",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("source_manifest_path=", result.stdout)
        stage_blocks_mock.assert_called_once()

    def test_acquire_blocks_invokes_composite_acquisition_once(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            write_config(config_path, output_root=tmp_path / "artifacts" / "baseline")
            raw_dir = (
                tmp_path
                / "artifacts"
                / "staging"
                / "alchemy"
                / "raw"
                / "ethereum"
                / "history"
            )
            enriched_dir = (
                tmp_path
                / "artifacts"
                / "staging"
                / "alchemy"
                / "enriched"
                / "ethereum"
                / "history"
            )
            report = RawPullValidationReport(
                dataset_path=raw_dir,
                expected_start_timestamp=1,
                expected_end_timestamp=2,
            )

            with patch(
                "spice_temporal.cli._acquire_blocks",
                return_value=SimpleNamespace(
                    raw=SimpleNamespace(
                        process=SimpleNamespace(stdout="", stderr=""),
                        validation=report,
                        source_manifest_path=source_manifest_path_for(raw_dir),
                    ),
                    enriched_output_dir=enriched_dir,
                    enriched_file_count=1,
                    enriched_source_manifest_path=source_manifest_path_for(enriched_dir),
                ),
            ) as acquire_blocks_mock:
                result = runner.invoke(
                    app,
                    [
                        "blocks",
                        "acquire",
                        str(config_path),
                        "ethereum",
                        "history",
                        "--pull-rpc-provider",
                        "alchemy",
                        "--enrich-rpc-provider",
                        "publicnode",
                        "--no-dry-run",
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("enriched_output_dir=", result.stdout)
        self.assertIn("enriched_source_manifest_path=", result.stdout)
        acquire_blocks_mock.assert_called_once()

    def test_promote_blocks_moves_staged_dataset_into_canonical_path(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            output_root = tmp_path / "artifacts"
            write_config(config_path, output_root=output_root)
            config = load_config(config_path)
            chain = config.chains[0]
            history = history_range_for_chain(chain)
            source_dir = tmp_path / "staging" / "publicnode" / "ethereum" / "history"
            source_dir.mkdir(parents=True)
            write_rows(
                source_dir / "ethereum__blocks__1_to_3.parquet",
                [
                    {
                        "block_number": 1,
                        "timestamp": history.start + 1,
                        "base_fee_per_gas": 100,
                        "gas_used": 15_000_001,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 2,
                        "timestamp": history.start + 13,
                        "base_fee_per_gas": 101,
                        "gas_used": 15_000_002,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 3,
                        "timestamp": history.start + 25,
                        "base_fee_per_gas": 102,
                        "gas_used": 15_000_003,
                        "chain_id": 1,
                    },
                ],
            )
            write_source_manifest(
                source_dir,
                config_path=config_path,
                chain=chain,
                segment=BlockSegment.HISTORY,
                timestamps=history,
                provider=RpcProvider(
                    name=RpcProviderName.PUBLICNODE,
                    urls={chain.name: "https://ethereum-rpc.publicnode.com"},
                    references={chain.name: "https://ethereum-rpc.publicnode.com"},
                ),
                pull=PullConfig(
                    requests_per_second=10,
                    max_concurrent_requests=2,
                    max_concurrent_chunks=1,
                ),
                overwrite=False,
                validation=None,
            )

            result = runner.invoke(
                app,
                [
                    "blocks",
                    "promote",
                    str(config_path),
                    "ethereum",
                    "history",
                    str(source_dir),
                ],
            )

            manifest_path = source_manifest_path_for(output_root / "raw" / "ethereum" / "history")
            manifest_exists = manifest_path.exists()

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("status=clean", result.stdout)
        self.assertTrue(manifest_exists)


if __name__ == "__main__":
    unittest.main()

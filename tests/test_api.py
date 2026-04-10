import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from spice_temporal._rpc import JsonRpcClient
from spice_temporal.api import (
    BlockPullResult,
    _acquire_blocks,
    _promote_block_pull,
    _pull_blocks,
    _stage_block_pull,
    load_artifact,
    load_config,
    run_simulation_workflow,
    run_training_workflow,
)
from spice_temporal.artifacts import SIMULATION_REPORT_FILENAME, TRAIN_REPORT_FILENAME
from spice_temporal.config import BlockSegment, PullConfig
from spice_temporal.cryo import history_range_for_chain
from spice_temporal.io import write_rows
from spice_temporal.provenance import source_manifest_path_for, write_source_manifest
from spice_temporal.raw_validation import RawPullValidationReport
from spice_temporal.rpc_providers import RpcProvider, RpcProviderName
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

    def test_pull_blocks_writes_dataset_level_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            output_root = tmp_path / "artifacts"
            write_config(config_path, output_root=output_root)

            with (
                patch.dict(
                    os.environ,
                    {"ETHEREUM_RPC_URL": "https://rpc.example.test"},
                    clear=False,
                ),
                patch(
                    "spice_temporal.api.run_cryo",
                    return_value=CompletedProcess(
                        args=["cryo"],
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                ),
            ):
                result = _pull_blocks(
                    config_path,
                    "ethereum",
                    "history",
                    rpc_provider=RpcProviderName.DIRECT,
                    dry_run=False,
                )

            manifest_path = source_manifest_path_for(output_root / "raw" / "ethereum" / "history")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.source_manifest_path, manifest_path)
            self.assertEqual(payload["chain"], "ethereum")
            self.assertEqual(payload["segment"], "history")
            self.assertEqual(payload["provider"], "direct")
            self.assertEqual(payload["provider_reference"], "$ETHEREUM_RPC_URL")
            self.assertIsNone(payload["validation"])

    def test_pull_blocks_source_manifest_includes_validation_summary_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            output_root = tmp_path / "artifacts"
            write_config(config_path, output_root=output_root)
            validation = RawPullValidationReport(
                dataset_path=output_root / "raw" / "ethereum" / "history",
                expected_start_timestamp=1,
                expected_end_timestamp=2,
                file_count=3,
                row_count=3000,
                status="warning",
                warnings=["edge rows outside requested range"],
            )

            with (
                patch.dict(
                    os.environ,
                    {"ETHEREUM_RPC_URL": "https://rpc.example.test"},
                    clear=False,
                ),
                patch(
                    "spice_temporal.api.run_cryo",
                    return_value=CompletedProcess(
                        args=["cryo"],
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                ),
                patch("spice_temporal.api._validate_dataset_path", return_value=validation),
            ):
                _pull_blocks(
                    config_path,
                    "ethereum",
                    "history",
                    rpc_provider=RpcProviderName.DIRECT,
                    dry_run=False,
                    validate_on_success=True,
                )

            manifest_path = source_manifest_path_for(output_root / "raw" / "ethereum" / "history")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["validation"]["status"], "warning")
            self.assertEqual(payload["validation"]["file_count"], 3)
            self.assertEqual(
                payload["validation"]["warnings"],
                ["edge rows outside requested range"],
            )

    def test_stage_block_pull_writes_into_provider_scoped_staging_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            output_root = tmp_path / "artifacts" / "baseline"
            write_config(config_path, output_root=output_root)

            with (
                patch.dict(
                    os.environ,
                    {"ETHEREUM_RPC_URL": "https://rpc.example.test"},
                    clear=False,
                ),
                patch(
                    "spice_temporal.api.run_cryo",
                    return_value=CompletedProcess(
                        args=["cryo"],
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                ),
            ):
                result = _stage_block_pull(
                    config_path,
                    "ethereum",
                    "history",
                    rpc_provider=RpcProviderName.DIRECT,
                    dry_run=False,
                )

            expected_dir = (
                output_root.parent
                / "staging"
                / "direct"
                / "raw"
                / "ethereum"
                / "history"
            )
            self.assertEqual(result.output_dir, expected_dir)
            self.assertEqual(
                result.source_manifest_path,
                source_manifest_path_for(expected_dir),
            )

    def test_acquire_blocks_supports_distinct_pull_and_enrich_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            output_root = tmp_path / "artifacts" / "baseline"
            write_config(config_path, output_root=output_root)
            staged_raw_dir = (
                output_root.parent
                / "staging"
                / "alchemy"
                / "raw"
                / "ethereum"
                / "history"
            )
            staged_raw_dir.mkdir(parents=True)
            write_rows(
                staged_raw_dir / "ethereum__blocks__1_to_3.parquet",
                [
                    {
                        "block_number": 1,
                        "timestamp": 1,
                        "base_fee_per_gas": 100,
                        "gas_used": 15_000_001,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 2,
                        "timestamp": 13,
                        "base_fee_per_gas": 101,
                        "gas_used": 15_000_002,
                        "chain_id": 1,
                    },
                    {
                        "block_number": 3,
                        "timestamp": 25,
                        "base_fee_per_gas": 102,
                        "gas_used": 15_000_003,
                        "chain_id": 1,
                    },
                ],
            )
            raw_manifest_path = source_manifest_path_for(staged_raw_dir)
            raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            raw_manifest_path.write_text("{}", encoding="utf-8")

            with (
                patch.dict(
                    os.environ,
                    {"ALCHEMY_API_KEY": "test-key"},
                    clear=False,
                ),
                patch(
                    "spice_temporal.api._execute_block_pull",
                    return_value=BlockPullResult(
                        output_dir=staged_raw_dir,
                        process=CompletedProcess(
                            args=["cryo"],
                            returncode=0,
                            stdout="",
                            stderr="",
                        ),
                        validation=None,
                        source_manifest_path=raw_manifest_path,
                    ),
                ),
                patch.object(
                    JsonRpcClient,
                    "get_block_gas_limits",
                    return_value={1: 30_000_001, 2: 30_000_002, 3: 30_000_003},
                ),
            ):
                result = _acquire_blocks(
                    config_path,
                    "ethereum",
                    "history",
                    pull_rpc_provider=RpcProviderName.ALCHEMY,
                    enrich_rpc_provider=RpcProviderName.PUBLICNODE,
                    dry_run=False,
                    validate_on_success=False,
                )

            enriched_dir = (
                output_root.parent
                / "staging"
                / "alchemy"
                / "enriched"
                / "ethereum"
                / "history"
            )
            self.assertEqual(result.enriched_output_dir, enriched_dir)
            self.assertEqual(result.enriched_file_count, 1)
            payload = json.loads(
                source_manifest_path_for(enriched_dir).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["provider"], "publicnode")
            self.assertEqual(payload["input_path"], str(staged_raw_dir.resolve()))
            self.assertEqual(
                payload["input_source_manifest_path"],
                str(raw_manifest_path.resolve()),
            )

    def test_promote_block_pull_validates_then_moves_into_canonical_baseline(self) -> None:
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

            provider = RpcProvider(
                name="publicnode",
                urls={chain.name: "https://ethereum-rpc.publicnode.com"},
                references={chain.name: "https://ethereum-rpc.publicnode.com"},
            )
            write_source_manifest(
                source_dir,
                config_path=config_path,
                chain=chain,
                segment=BlockSegment.HISTORY,
                timestamps=history,
                provider=provider,
                pull=PullConfig(
                    requests_per_second=10,
                    max_concurrent_requests=2,
                    max_concurrent_chunks=1,
                ),
                overwrite=False,
                validation=None,
            )

            result = _promote_block_pull(
                config_path,
                "ethereum",
                "history",
                source_dir,
            )

            destination_dir = output_root / "raw" / "ethereum" / "history"
            self.assertEqual(result.output_dir, destination_dir.resolve())
            self.assertFalse(source_dir.exists())
            self.assertTrue(destination_dir.exists())
            payload = json.loads(
                source_manifest_path_for(destination_dir).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["provider"], "publicnode")
            self.assertEqual(payload["output_dir"], str(destination_dir.resolve()))
            self.assertEqual(
                payload["promotion"]["promoted_from"],
                str(source_dir.resolve()),
            )
            self.assertEqual(payload["validation"]["status"], "clean")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys

import polars as pl
import pytest

from spice.acquisition.cryo import TimestampRange, run_cryo
from spice.acquisition.enrich import enrich_frame_with_gas_limit, enrich_path
from spice.acquisition.provenance import (
    load_source_manifest,
    source_manifest_path_for,
    update_source_manifest_for_promotion,
    write_source_manifest,
)
from spice.acquisition.raw_validation import RawPullValidationReport
from spice.acquisition.rpc import JsonRpcClient
from spice.acquisition.rpc_providers import RpcProviderName, resolve_rpc_provider
from spice.core.config import BlockSegment, ChainConfig, ChainName, PullConfig
from spice.core.console import NullReporter
from spice.data.block_schema import ENRICHED_BLOCK_SCHEMA
from tests.support import make_block_rows, make_history_rows, write_dataset_dir


def test_enrich_frame_with_gas_limit_fills_missing_blocks() -> None:
    def fetch_gas_limits(block_numbers: list[int]) -> dict[int, int]:
        return {block: 30_000_000 + block for block in block_numbers}

    frame = pl.DataFrame(
        make_block_rows(
            4,
            start_block=1,
            start_timestamp=1_700_000_000,
            include_gas_limit=True,
            missing_gas_limit_blocks={2, 4},
        )
    )

    enriched, fetched = enrich_frame_with_gas_limit(
        frame,
        fetch_gas_limits=fetch_gas_limits,
        batch_size=2,
        max_methods_per_second=1_000.0,
    )

    assert fetched == 2
    assert enriched.schema == ENRICHED_BLOCK_SCHEMA
    assert enriched["gas_limit"].to_list() == [30_000_000, 30_000_002, 30_000_000, 30_000_004]


def test_enrich_frame_with_gas_limit_canonicalizes_wide_uint_input() -> None:
    def fetch_gas_limits(block_numbers: list[int]) -> dict[int, int]:
        return {block: 45_000_000 for block in block_numbers}

    frame = pl.DataFrame(
        {
            "block_hash": [b"a", b"b"],
            "block_number": pl.Series([1, 2], dtype=pl.UInt32),
            "timestamp": pl.Series([1_700_000_000, 1_700_000_002], dtype=pl.UInt32),
            "base_fee_per_gas": pl.Series([3_000_000_000, 3_100_000_000], dtype=pl.UInt64),
            "gas_used": pl.Series([20_000_000, 21_000_000], dtype=pl.UInt64),
            "chain_id": pl.Series([137, 137], dtype=pl.UInt64),
            "gas_limit": pl.Series([None, None], dtype=pl.Int64),
        }
    )

    enriched, fetched = enrich_frame_with_gas_limit(
        frame,
        fetch_gas_limits=fetch_gas_limits,
        batch_size=2,
        max_methods_per_second=1_000.0,
    )

    assert fetched == 2
    assert enriched.schema == ENRICHED_BLOCK_SCHEMA
    assert enriched.columns == list(ENRICHED_BLOCK_SCHEMA)


def test_enrich_path_writes_parquet_outputs(tmp_path) -> None:
    def fetch_gas_limits(block_numbers: list[int]) -> dict[int, int]:
        return {block: 31_000_000 + block for block in block_numbers}

    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "enriched"
    write_dataset_dir(
        input_dir,
        make_block_rows(
            5,
            start_block=1,
            start_timestamp=1_700_000_000,
            include_gas_limit=True,
            missing_gas_limit_blocks={1, 3},
        ),
    )

    written = enrich_path(
        input_dir,
        output_dir,
        fetch_gas_limits=fetch_gas_limits,
        batch_size=2,
        max_methods_per_second=1_000.0,
    )

    assert len(written) == 1
    assert written[0].is_file()
    written_frame = pl.read_parquet(written[0])
    assert written_frame.schema == ENRICHED_BLOCK_SCHEMA
    assert written_frame["gas_limit"].null_count() == 0


def test_json_rpc_client_retries_throttled_items(monkeypatch) -> None:
    client = JsonRpcClient("https://rpc.example.test", max_retries=2, retry_backoff_seconds=0.0)
    responses = iter(
        [
            [
                {"id": 1, "result": {"number": "0x1", "gasLimit": "0x64"}},
                {"id": 2, "error": {"code": 429}},
            ],
            [
                {"id": 1, "result": {"number": "0x2", "gasLimit": "0x65"}},
            ],
        ]
    )
    monkeypatch.setattr(client, "_post", lambda payload: next(responses))
    monkeypatch.setattr("spice.acquisition.rpc.time.sleep", lambda _: None)

    gas_limits = client.get_block_gas_limits([1, 2])

    assert gas_limits == {1: 100, 2: 101}


def test_promotion_requires_existing_raw_manifest(tmp_path) -> None:
    dataset_dir = tmp_path / "raw"
    write_dataset_dir(dataset_dir, make_history_rows(16))

    with pytest.raises(ValueError, match="requires an existing raw source manifest"):
        update_source_manifest_for_promotion(
            dataset_dir,
            promoted_from=tmp_path / "staging",
            validation=RawPullValidationReport(
                dataset_path=dataset_dir,
                expected_start_timestamp=0,
                expected_end_timestamp=1,
            ),
        )


def test_source_manifest_round_trip(tmp_path) -> None:
    dataset_dir = tmp_path / "raw"
    dataset_dir.mkdir()
    provider = resolve_rpc_provider(RpcProviderName.PUBLICNODE, chains=(ChainName.ETHEREUM,))
    manifest_path = write_source_manifest(
        dataset_dir,
        config_path=None,
        chain=ChainConfig(
            name=ChainName.ETHEREUM,
            chain_id=1,
            block_time_seconds=12.0,
            history_days=1,
        ),
        segment=BlockSegment.HISTORY,
        timestamps=TimestampRange(start=1, end=10),
        provider=provider,
        pull=PullConfig(),
        overwrite=False,
        validation=None,
    )

    loaded = load_source_manifest(dataset_dir)

    assert manifest_path == source_manifest_path_for(dataset_dir)
    assert loaded is not None
    assert loaded.chain == "ethereum"
    assert loaded.provider == "publicnode"


def test_run_cryo_polls_progress_before_stdout_lines(tmp_path, monkeypatch) -> None:
    class RecordingReporter(NullReporter):
        def __init__(self) -> None:
            self.pull_updates: list[tuple[int, int | None, str | None]] = []

        def update_pull(
            self,
            *,
            completed_chunks: int,
            total_chunks: int | None,
            latest_output: str | None = None,
        ) -> None:
            self.pull_updates.append((completed_chunks, total_chunks, latest_output))

    output_dir = tmp_path / "raw"
    reporter = RecordingReporter()
    provider = resolve_rpc_provider(RpcProviderName.PUBLICNODE, chains=(ChainName.ETHEREUM,))
    script = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "import time",
            "output_dir = Path(sys.argv[1])",
            "output_dir.mkdir(parents=True, exist_ok=True)",
            "time.sleep(0.05)",
            "(output_dir / 'ethereum__blocks__1_to_1.parquet').write_text('ok', encoding='utf-8')",
            "time.sleep(0.15)",
            "print('done', flush=True)",
        ]
    )

    monkeypatch.setattr(
        "spice.acquisition.cryo.build_cryo_args",
        lambda *args, **kwargs: [sys.executable, "-c", script, str(output_dir)],
    )
    monkeypatch.setattr(
        "spice.acquisition.cryo.build_cryo_command",
        lambda *args, **kwargs: f"{sys.executable} -c <script>",
    )
    monkeypatch.setattr("spice.acquisition.cryo.CRYO_PROGRESS_POLL_INTERVAL_SECONDS", 0.02)

    result = run_cryo(
        ChainConfig(
            name=ChainName.ETHEREUM,
            chain_id=1,
            block_time_seconds=12.0,
            history_days=1,
        ),
        PullConfig(),
        output_dir,
        TimestampRange(start=1, end=13),
        provider=provider,
        reporter=reporter,
    )

    done_index = next(
        index
        for index, (_completed, _total, latest_output) in enumerate(reporter.pull_updates)
        if latest_output == "done"
    )
    progress_index = next(
        index
        for index, (completed, _total, latest_output) in enumerate(reporter.pull_updates)
        if completed == 1 and latest_output is None
    )

    assert progress_index < done_index
    assert result.completed_chunks == 1

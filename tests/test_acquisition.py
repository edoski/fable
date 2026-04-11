from __future__ import annotations

import json
import sys
from typing import cast

import polars as pl
import pytest

from spice.acquisition.cryo import (
    CryoRunResult,
    TimestampRange,
    history_range_for_chain,
    run_cryo,
)
from spice.acquisition.enrich import enrich_frame_with_gas_limit, enrich_path
from spice.acquisition.raw_normalization import normalize_raw_dataset
from spice.core.config import ChainConfig, ChainName, ProviderConfig, PullConfig, RpcProviderName
from spice.core.console import NullReporter
from spice.data.block_schema import ENRICHED_BLOCK_SCHEMA
from spice.data.io import load_enriched_block_frame, read_block_dataset
from spice.workflows.acquire import run as run_acquire
from tests.support import (
    base_overrides,
    compose_experiment,
    make_block_rows,
    write_dataset_dir,
    write_raw_chunk,
)


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
    provider = ProviderConfig(
        name=RpcProviderName.PUBLICNODE,
        endpoints={"ethereum": "https://rpc.example.test"},
        references={"ethereum": "https://rpc.example.test"},
    )
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
        lambda *_args, **_kwargs: [sys.executable, "-c", script, str(output_dir)],
    )
    monkeypatch.setattr(
        "spice.acquisition.cryo.build_cryo_command",
        lambda *_args, **_kwargs: "python fake_cryo.py",
    )

    result = run_cryo(
        ChainConfig(name=ChainName.ETHEREUM, chain_id=1, block_time_seconds=12.0, history_days=1),
        PullConfig(chunk_size=1),
        output_dir,
        TimestampRange(start=1, end=13),
        provider=provider,
        reporter=reporter,
    )

    assert result.completed_chunks == 1
    assert reporter.pull_updates
    assert reporter.pull_updates[0][0] == 1
    assert any(update[2] == "done" for update in reporter.pull_updates)


def test_normalize_raw_dataset_trims_edge_rows_and_rechunks(tmp_path) -> None:
    scratch_dir = tmp_path / "scratch"
    output_dir = tmp_path / "raw"
    rows = make_block_rows(
        8,
        start_block=1,
        start_timestamp=99,
        block_time_seconds=1,
        include_gas_limit=False,
    )
    write_dataset_dir(scratch_dir, rows)

    written = normalize_raw_dataset(
        scratch_dir,
        output_dir,
        chain_name="ethereum",
        expected_chain_id=1,
        expected_start_timestamp=100,
        expected_end_timestamp=106,
        chunk_size=4,
    )

    assert [path.name for path in written] == [
        "ethereum__blocks__2_to_5.parquet",
        "ethereum__blocks__6_to_7.parquet",
    ]
    frame = read_block_dataset(output_dir).sort("block_number")
    assert frame["block_number"].to_list() == [2, 3, 4, 5, 6, 7]
    assert frame["timestamp"].to_list() == [100, 101, 102, 103, 104, 105]


def test_normalize_raw_dataset_rejects_internal_out_of_window_rows(tmp_path) -> None:
    scratch_dir = tmp_path / "scratch"
    rows = [
        {
            "block_number": 1,
            "timestamp": 100,
            "base_fee_per_gas": 1,
            "gas_used": 1,
            "chain_id": 1,
        },
        {
            "block_number": 2,
            "timestamp": 101,
            "base_fee_per_gas": 1,
            "gas_used": 1,
            "chain_id": 1,
        },
        {
            "block_number": 3,
            "timestamp": 99,
            "base_fee_per_gas": 1,
            "gas_used": 1,
            "chain_id": 1,
        },
        {
            "block_number": 4,
            "timestamp": 102,
            "base_fee_per_gas": 1,
            "gas_used": 1,
            "chain_id": 1,
        },
    ]
    write_dataset_dir(scratch_dir, cast(list[dict[str, int | None]], rows))

    with pytest.raises(ValueError, match="inside the requested block window"):
        normalize_raw_dataset(
            scratch_dir,
            tmp_path / "raw",
            chain_name="ethereum",
            expected_chain_id=1,
            expected_start_timestamp=100,
            expected_end_timestamp=103,
            chunk_size=1000,
        )


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (
            [
                {
                    "block_number": 1,
                    "timestamp": 100,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 1,
                },
                {
                    "block_number": 1,
                    "timestamp": 101,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 1,
                },
            ],
            "duplicate block_number",
        ),
        (
            [
                {
                    "block_number": 1,
                    "timestamp": 100,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 1,
                },
                {
                    "block_number": 3,
                    "timestamp": 101,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 1,
                },
            ],
            "non-contiguous block_number",
        ),
        (
            [
                {
                    "block_number": 1,
                    "timestamp": 100,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 137,
                },
                {
                    "block_number": 2,
                    "timestamp": 101,
                    "base_fee_per_gas": 1,
                    "gas_used": 1,
                    "chain_id": 137,
                },
            ],
            "chain_id mismatch",
        ),
    ],
)
def test_normalize_raw_dataset_rejects_invalid_sequences(tmp_path, rows, match) -> None:
    scratch_dir = tmp_path / "scratch"
    write_dataset_dir(scratch_dir, cast(list[dict[str, int | None]], rows))

    with pytest.raises(ValueError, match=match):
        normalize_raw_dataset(
            scratch_dir,
            tmp_path / "raw",
            chain_name="ethereum",
            expected_chain_id=1,
            expected_start_timestamp=100,
            expected_end_timestamp=103,
            chunk_size=1000,
        )


def test_acquire_workflow_writes_validation_reports(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "pull.dry_run=false"],
    )

    def fake_run_cryo(chain, _pull, output_dir, timestamps, **_kwargs):
        segment = output_dir.name
        rows = make_block_rows(
            4,
            start_block=1 if segment == "history" else 10_001,
            start_timestamp=timestamps.start - int(chain.block_time_seconds),
            block_time_seconds=int(chain.block_time_seconds),
            include_gas_limit=False,
        )
        write_raw_chunk(output_dir, chain_name=chain.name.value, rows=rows)
        return CryoRunResult(command=f"cryo {segment}", completed_chunks=1, expected_chunks=1)

    class FakeBlockClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
            return {block_number: 30_000_000 for block_number in block_numbers}

    monkeypatch.setattr("spice.workflows.acquire.run_cryo", fake_run_cryo)
    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeBlockClient)

    run_acquire(config, reporter=NullReporter())

    metadata_dir = tmp_path / "artifacts" / "datasets" / "ethereum" / ".spice"
    metadata_path = metadata_dir / "metadata.json"
    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["validation"]["raw"]["history"]["status"] == "clean"
    assert payload["validation"]["enriched"]["evaluation"]["status"] == "clean"
    assert "issues" not in payload["validation"]["raw"]["history"]
    history_dir = tmp_path / "artifacts" / "datasets" / "ethereum" / "enriched" / "history"
    history_frame = load_enriched_block_frame(history_dir)
    assert history_frame.height > 0
    assert int(history_frame["timestamp"][0]) == history_range_for_chain(config.chain).start


def test_acquire_workflow_rejects_non_trim_boundary_violations(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "pull.dry_run=false"],
    )

    def fake_run_cryo(chain, _pull, output_dir, timestamps, **_kwargs):
        rows = [
            {
                "block_number": 1,
                "timestamp": timestamps.start,
                "base_fee_per_gas": 1,
                "gas_used": 1,
                "chain_id": chain.chain_id,
            },
            {
                "block_number": 2,
                "timestamp": timestamps.start + int(chain.block_time_seconds),
                "base_fee_per_gas": 1,
                "gas_used": 1,
                "chain_id": chain.chain_id,
            },
            {
                "block_number": 3,
                "timestamp": timestamps.start - int(chain.block_time_seconds),
                "base_fee_per_gas": 1,
                "gas_used": 1,
                "chain_id": chain.chain_id,
            },
            {
                "block_number": 4,
                "timestamp": timestamps.start + 2 * int(chain.block_time_seconds),
                "base_fee_per_gas": 1,
                "gas_used": 1,
                "chain_id": chain.chain_id,
            },
        ]
        write_dataset_dir(output_dir, rows)
        return CryoRunResult(command="cryo invalid", completed_chunks=1, expected_chunks=1)

    class FakeBlockClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
            return {block_number: 30_000_000 for block_number in block_numbers}

    monkeypatch.setattr("spice.workflows.acquire.run_cryo", fake_run_cryo)
    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeBlockClient)

    with pytest.raises(ValueError, match="inside the requested block window"):
        run_acquire(config, reporter=NullReporter())

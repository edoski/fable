from __future__ import annotations

import json
import sys
from typing import cast

import polars as pl
import pytest

from spice.acquisition.cryo import (
    CryoRunResult,
    TimestampRange,
    run_cryo,
)
from spice.acquisition.enrich import enrich_frame_with_gas_limit
from spice.acquisition.metadata import load_dataset_metadata
from spice.acquisition.raw_normalization import normalize_raw_dataset
from spice.acquisition.windowing import history_range_from_metadata
from spice.core.config import RpcProviderName
from spice.core.console import NullReporter
from spice.data.block_schema import ENRICHED_BLOCK_SCHEMA
from spice.data.io import load_enriched_block_frame, read_block_dataset
from spice.workflows.acquire import run as run_acquire
from tests.support import (
    base_overrides,
    compose_experiment,
    make_acquisition_config,
    make_block_rows,
    make_chain_config,
    make_provider_config,
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


def test_run_cryo_polls_progress_before_stdout_lines(tmp_path, monkeypatch) -> None:
    class RecordingReporter(NullReporter):
        def __init__(self) -> None:
            self.events: list[tuple[str, int | str | None]] = []

        def start_task(
            self,
            name: str,
            *,
            total: int | None = None,
            unit: str | None = None,
        ) -> int:
            self.events.append(("start", total))
            return 1

        def update_task(
            self,
            task_id: int,
            *,
            completed: int | None = None,
            advance: int | None = None,
            message: str | None = None,
        ) -> None:
            self.events.append(("update", completed))

        def throttled_log(
            self,
            key: str,
            message: str,
            *,
            interval_seconds: float = 10.0,
            level: str = "info",
        ) -> None:
            self.events.append(("log", message))

    output_dir = tmp_path / "raw"
    reporter = RecordingReporter()
    provider = make_provider_config(
        name=RpcProviderName.PUBLICNODE,
        endpoint="https://rpc.example.test",
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
        make_chain_config(),
        make_acquisition_config(),
        output_dir,
        TimestampRange(start=1, end=13),
        provider=provider,
        reporter=reporter,
    )

    assert result.completed_chunks == 1
    assert reporter.events
    assert ("update", 1) in reporter.events
    assert ("log", "done") in reporter.events


class _FakeBlockClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
        return {block_number: 30_000_000 for block_number in block_numbers}


def _install_clean_acquire_fakes(monkeypatch) -> None:
    def fake_run_cryo(chain, _pull, output_dir, timestamps, **_kwargs):
        segment = output_dir.name
        block_time_seconds = int(chain.block_time_seconds)
        row_count = max(1, (timestamps.end - timestamps.start) // block_time_seconds) + 2
        rows = make_block_rows(
            row_count,
            start_block=1 if segment == "history" else 10_001,
            start_timestamp=timestamps.start - block_time_seconds,
            block_time_seconds=block_time_seconds,
            include_gas_limit=False,
        )
        write_raw_chunk(output_dir, chain_name=chain.name.value, rows=rows)
        return CryoRunResult(command=f"cryo {segment}", completed_chunks=1, expected_chunks=1)

    monkeypatch.setattr("spice.acquisition.datasets.run_cryo", fake_run_cryo)
    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", _FakeBlockClient)


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
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "acquisition.dry_run=false"],
    )
    _install_clean_acquire_fakes(monkeypatch)

    run_acquire(config, reporter=NullReporter())

    metadata_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / ".spice"
    )
    metadata_path = metadata_dir / "metadata.json"
    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = load_dataset_metadata(metadata_path)
    assert metadata is not None
    history_window = history_range_from_metadata(metadata)
    assert payload["dataset"]["id"] == "icdcs_2025_11_09"
    assert metadata.dataset.id == "icdcs_2025_11_09"
    assert payload["validation"]["raw"]["history"]["status"] == "clean"
    assert payload["validation"]["enriched"]["evaluation"]["status"] == "clean"
    assert "issues" not in payload["validation"]["raw"]["history"]
    history_dir = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / "enriched"
        / "history"
    )
    history_frame = load_enriched_block_frame(history_dir)
    assert history_frame.height > 0
    assert int(history_frame["timestamp"][0]) == payload["windows"]["history"]["start_timestamp"]
    assert int(history_frame["timestamp"][0]) == history_window.start


def test_acquire_reuses_larger_valid_dataset_for_lower_target(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "acquisition.dry_run=false"],
    )
    _install_clean_acquire_fakes(monkeypatch)
    run_acquire(config, reporter=NullReporter())

    lower_config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "provider=publicnode",
            "acquisition.dry_run=false",
            "dataset.sampling.anchor_count=8",
            "dataset.sampling.history_anchor_count=8",
        ],
    )
    monkeypatch.setattr(
        "spice.acquisition.datasets.run_cryo",
        lambda *_args, **_kwargs: pytest.fail("existing dataset should have been reused"),
    )
    run_acquire(lower_config, reporter=NullReporter())

    metadata_path = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / ".spice"
        / "metadata.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["validation"]["raw"]["history"]["rows"] > 8


def test_acquire_rejects_metadata_mismatch_without_overwrite(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "acquisition.dry_run=false"],
    )
    metadata_path = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / ".spice"
        / "metadata.json"
    )
    _install_clean_acquire_fakes(monkeypatch)
    run_acquire(config, reporter=NullReporter())

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["windows"]["evaluation"]["start_timestamp"] += 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata does not match"):
        run_acquire(config, reporter=NullReporter())


def test_acquire_expands_short_history_window_backward(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "provider=publicnode",
            "acquisition.dry_run=false",
            "acquisition.raw.chunk_size=5",
            "dataset.sampling.anchor_count=20",
            "dataset.sampling.history_anchor_count=20",
        ],
    )
    history_starts: list[int] = []

    def fake_run_cryo(chain, _pull, output_dir, timestamps, **_kwargs):
        block_time_seconds = int(chain.block_time_seconds)
        if output_dir.name == "history":
            history_starts.append(timestamps.start)
            row_count = 3 if len(history_starts) == 1 else 300
            start_timestamp = timestamps.end - row_count * block_time_seconds
        else:
            row_count = 10
            start_timestamp = timestamps.start
        rows = make_block_rows(
            row_count,
            start_block=1 if output_dir.name == "history" else 10_001,
            start_timestamp=start_timestamp,
            block_time_seconds=block_time_seconds,
            include_gas_limit=False,
        )
        write_dataset_dir(output_dir, rows)
        return CryoRunResult(
            command=f"cryo {output_dir.name}",
            completed_chunks=1,
            expected_chunks=1,
        )

    monkeypatch.setattr("spice.acquisition.datasets.run_cryo", fake_run_cryo)
    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", _FakeBlockClient)

    run_acquire(config, reporter=NullReporter())

    assert len(history_starts) == 2
    assert history_starts[1] < history_starts[0]
    metadata_path = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "ethereum"
        / "icdcs_2025_11_09"
        / ".spice"
        / "metadata.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["windows"]["history"]["start_timestamp"] == history_starts[1]


def test_acquire_workflow_rejects_non_trim_boundary_violations(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["provider=publicnode", "acquisition.dry_run=false"],
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

    monkeypatch.setattr("spice.acquisition.datasets.run_cryo", fake_run_cryo)
    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FakeBlockClient)

    with pytest.raises(ValueError, match="inside the requested block window"):
        run_acquire(config, reporter=NullReporter())

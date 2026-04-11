import polars as pl
import pytest
from pandera.errors import SchemaError, SchemaErrors

from spice.acquisition.raw_validation import validate_raw_pull
from spice.core.config import (
    ChainConfig,
    ChainName,
    ModelConfig,
    ModelFamily,
    SplitConfig,
    TrainingConfig,
)
from spice.data.block_schema import ENRICHED_BLOCK_SCHEMA
from spice.data.datasets import derive_dataset_geometry
from spice.data.io import iter_block_files, load_enriched_block_frame
from spice.data.validation import validate_exact_window_dataset
from spice.modeling.pipeline import (
    TrainingSpec,
    prepare_inference_dataset,
    prepare_training_dataset,
)
from tests.support import (
    make_evaluation_rows,
    make_history_rows,
    write_dataset_dir,
    write_raw_chunk,
)


def _timestamp(row: dict[str, int | None]) -> int:
    value = row["timestamp"]
    assert value is not None
    return int(value)


def _set_timestamp(row: dict[str, int | None], value: int) -> None:
    row["timestamp"] = value


def test_iter_block_files_ignores_hidden_metadata(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    write_dataset_dir(dataset_dir, make_history_rows(32))
    hidden_manifest = dataset_dir / ".spice" / "metadata.json"
    hidden_manifest.parent.mkdir(parents=True)
    hidden_manifest.write_text("{}", encoding="utf-8")

    files = iter_block_files(dataset_dir)

    assert len(files) == 1
    assert files[0].name == "blocks.parquet"


def test_load_enriched_block_frame_rejects_duplicate_block_numbers(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(8)
    write_dataset_dir(dataset_dir, rows[:4] + [rows[3]] + rows[4:])

    with pytest.raises((ValueError, SchemaError, SchemaErrors)):
        load_enriched_block_frame(dataset_dir)


def test_load_enriched_block_frame_rejects_noncanonical_schema(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    frame = make_history_rows(4)
    # Simulate a provider-shaped "enriched" dataset that was never canonicalized.

    wide_frame = pl.DataFrame(
        {
            "block_hash": [b"a", b"b", b"c", b"d"],
            "block_number": pl.Series([row["block_number"] for row in frame], dtype=pl.UInt32),
            "timestamp": pl.Series([row["timestamp"] for row in frame], dtype=pl.UInt32),
            "base_fee_per_gas": pl.Series(
                [row["base_fee_per_gas"] for row in frame], dtype=pl.UInt64
            ),
            "gas_used": pl.Series([row["gas_used"] for row in frame], dtype=pl.UInt64),
            "chain_id": pl.Series([row["chain_id"] for row in frame], dtype=pl.UInt64),
            "gas_limit": pl.Series([row["gas_limit"] for row in frame], dtype=pl.Int64),
        }
    )
    wide_frame.write_parquet(dataset_dir / "blocks.parquet")

    with pytest.raises((ValueError, SchemaError, SchemaErrors)):
        load_enriched_block_frame(dataset_dir)


def test_validate_raw_pull_detects_file_range_gaps(tmp_path) -> None:
    dataset_dir = tmp_path / "raw"
    write_raw_chunk(
        dataset_dir,
        chain_name="ethereum",
        rows=make_history_rows(3),
    )
    write_raw_chunk(
        dataset_dir,
        chain_name="ethereum",
        rows=make_evaluation_rows(2, start_block=5),
    )

    report = validate_raw_pull(
        dataset_dir,
        expected_chain_name="ethereum",
        expected_chain_id=1,
        expected_start_timestamp=0,
        expected_end_timestamp=2_000_000_000,
        expected_chunk_size=1000,
    )

    assert report.status == "error"
    assert report.gap_count >= 1


def test_validate_exact_window_dataset_passes_on_canonical_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(8)
    write_dataset_dir(dataset_dir, rows)

    report = validate_exact_window_dataset(
        dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "clean"
    assert report.row_count == 8
    assert report.duplicate_count == 0
    assert report.gap_count == 0


def test_validate_exact_window_dataset_rejects_duplicate_blocks(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(4)
    write_dataset_dir(dataset_dir, rows[:3] + [rows[2]])

    report = validate_exact_window_dataset(
        dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "error"
    assert report.duplicate_count == 1


def test_validate_exact_window_dataset_rejects_gaps(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(4)
    write_dataset_dir(dataset_dir, [rows[0], rows[1], rows[3]])

    report = validate_exact_window_dataset(
        dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "error"
    assert report.gap_count == 1


def test_validate_exact_window_dataset_rejects_out_of_range_timestamps(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(4)
    shifted = [dict(row) for row in rows]
    _set_timestamp(shifted[0], _timestamp(shifted[0]) - 12)
    write_dataset_dir(dataset_dir, shifted)

    report = validate_exact_window_dataset(
        dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "error"
    assert report.below_start_count == 1


def test_validate_exact_window_dataset_rejects_chain_mismatch(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(4)
    shifted = [dict(row) for row in rows]
    for row in shifted:
        row["chain_id"] = 137
    write_dataset_dir(dataset_dir, shifted)

    report = validate_exact_window_dataset(
        dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "error"
    assert report.chain_id == 137


def test_prepare_training_and_inference_datasets(tmp_path) -> None:
    history_dir = tmp_path / "history"
    evaluation_dir = tmp_path / "evaluation"
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())

    history_blocks = load_enriched_block_frame(history_dir)
    evaluation_blocks = load_enriched_block_frame(evaluation_dir)
    assert history_blocks.schema == ENRICHED_BLOCK_SCHEMA
    assert evaluation_blocks.schema == ENRICHED_BLOCK_SCHEMA
    spec = TrainingSpec(
        chain=ChainConfig(
            name=ChainName.ETHEREUM,
            chain_id=1,
            block_time_seconds=12.0,
            history_days=1,
        ),
        model=ModelConfig(family=ModelFamily.LSTM),
        max_delay_seconds=36,
        lookback_seconds=120,
        target_anchor_count=48,
        split=SplitConfig(train_fraction=0.7, validation_fraction=0.15),
        training=TrainingConfig(max_epochs=1, effective_batch_size=8, device="cpu"),
    )

    prepared = prepare_training_dataset(history_blocks, spec=spec)
    inference = prepare_inference_dataset(
        history_blocks,
        evaluation_blocks,
        geometry=derive_dataset_geometry(
            lookback_seconds=120,
            max_delay_seconds=36,
            block_time_seconds=12.0,
        ),
        scaler=prepared.scaler,
    )

    assert prepared.n_examples_total == 48
    assert prepared.store.n_features > 0
    assert prepared.split_indices.test.size > 0
    assert inference.n_examples_total > 0
    assert inference.store.n_features == prepared.store.n_features

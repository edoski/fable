import polars as pl
import pytest
from pandera.errors import SchemaError, SchemaErrors

from spice.core.config import SplitConfig
from spice.data.block_contract import BLOCK_SCHEMA
from spice.data.datasets import derive_dataset_geometry
from spice.data.io import iter_block_files, load_block_frame, read_block_dataset
from spice.data.validation import validate_exact_window_frame
from spice.modeling.pipeline import (
    TrainingSpec,
    prepare_inference_dataset,
    prepare_training_dataset,
)
from tests.support import (
    TEST_WINDOW_START_TIMESTAMP,
    make_chain_config,
    make_evaluation_rows,
    make_history_rows,
    make_model_config,
    make_training_config,
    write_dataset_dir,
)


def _timestamp(row: dict[str, int | None]) -> int:
    value = row["timestamp"]
    assert value is not None
    return int(value)


def test_iter_block_files_ignores_hidden_metadata(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    write_dataset_dir(dataset_dir, make_history_rows(32))
    hidden_manifest = dataset_dir / ".spice" / "metadata.json"
    hidden_manifest.parent.mkdir(parents=True)
    hidden_manifest.write_text("{}", encoding="utf-8")

    files = iter_block_files(dataset_dir)

    assert len(files) == 1
    assert files[0].name == "blocks.parquet"


def test_load_block_frame_rejects_duplicate_block_numbers(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(8)
    write_dataset_dir(dataset_dir, rows[:4] + [rows[3]] + rows[4:])

    with pytest.raises((ValueError, SchemaError, SchemaErrors)):
        load_block_frame(dataset_dir)


def test_load_block_frame_rejects_noncanonical_schema(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    frame = make_history_rows(4)
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
        load_block_frame(dataset_dir)


def test_validate_exact_window_dataset_passes_on_canonical_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(8)
    write_dataset_dir(dataset_dir, rows)
    frame = load_block_frame(dataset_dir)

    report = validate_exact_window_frame(
        frame,
        dataset_path=dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "clean"
    assert report.row_count == 8
    assert report.duplicate_count == 0
    assert report.gap_count == 0


@pytest.mark.parametrize(
    ("mutate_rows", "assert_report"),
    [
        (
            lambda rows: rows[:3] + [rows[2]],
            lambda report: report.duplicate_count == 1,
        ),
        (
            lambda rows: [rows[0], rows[1], rows[3]],
            lambda report: report.gap_count == 1,
        ),
        (
            lambda rows: [
                {**row, "timestamp": _timestamp(row) - 12} if index == 0 else dict(row)
                for index, row in enumerate(rows)
            ],
            lambda report: report.below_start_count == 1,
        ),
        (
            lambda rows: [{**row, "chain_id": 137} for row in rows],
            lambda report: report.chain_id == 137,
        ),
    ],
)
def test_validate_exact_window_dataset_rejects_invalid_inputs(
    tmp_path,
    mutate_rows,
    assert_report,
) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(4)
    write_dataset_dir(dataset_dir, mutate_rows(rows))
    frame = read_block_dataset(dataset_dir)

    report = validate_exact_window_frame(
        frame,
        dataset_path=dataset_dir,
        expected_chain_id=1,
        expected_start_timestamp=_timestamp(rows[0]),
        expected_end_timestamp=_timestamp(rows[-1]) + 12,
    )

    assert report.status == "error"
    assert assert_report(report)


def test_prepare_training_and_inference_datasets(tmp_path) -> None:
    history_dir = tmp_path / "history"
    evaluation_dir = tmp_path / "evaluation"
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())

    history_blocks = load_block_frame(history_dir)
    evaluation_blocks = load_block_frame(evaluation_dir)
    assert history_blocks.schema == BLOCK_SCHEMA
    assert evaluation_blocks.schema == BLOCK_SCHEMA
    spec = TrainingSpec(
        chain=make_chain_config(),
        dataset_id="icdcs_2025_11_09",
        model=make_model_config(),
        max_delay_seconds=36,
        lookback_seconds=120,
        anchor_count=48,
        split=SplitConfig(train_fraction=0.7, validation_fraction=0.15),
        training=make_training_config(),
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
        window_start_timestamp=TEST_WINDOW_START_TIMESTAMP,
        window_end_timestamp=TEST_WINDOW_START_TIMESTAMP + 180 * 12,
    )

    assert prepared.n_examples_total == 48
    assert prepared.store.n_features > 0
    assert prepared.split_indices.test.size > 0
    assert inference.n_examples_total > 0
    assert inference.store.n_features == prepared.store.n_features

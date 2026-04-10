import pytest

from spice.acquisition.raw_validation import validate_raw_pull
from spice.core.config import (
    ChainConfig,
    ChainName,
    ModelConfig,
    ModelFamily,
    SplitConfig,
    TrainingConfig,
)
from spice.data.datasets import derive_dataset_geometry
from spice.data.io import iter_block_files, load_enriched_block_frame
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


def test_iter_block_files_ignores_hidden_metadata(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    write_dataset_dir(dataset_dir, make_history_rows(32))
    hidden_manifest = dataset_dir / ".spice" / "source.json"
    hidden_manifest.parent.mkdir(parents=True)
    hidden_manifest.write_text("{}", encoding="utf-8")

    files = iter_block_files(dataset_dir)

    assert len(files) == 1
    assert files[0].name == "blocks.parquet"


def test_load_enriched_block_frame_rejects_duplicate_block_numbers(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = make_history_rows(8)
    write_dataset_dir(dataset_dir, rows[:4] + [rows[3]] + rows[4:])

    with pytest.raises(ValueError, match="duplicate block_number"):
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


def test_prepare_training_and_inference_datasets(tmp_path) -> None:
    history_dir = tmp_path / "history"
    evaluation_dir = tmp_path / "evaluation"
    write_dataset_dir(history_dir, make_history_rows())
    write_dataset_dir(evaluation_dir, make_evaluation_rows())

    history_blocks = load_enriched_block_frame(history_dir)
    evaluation_blocks = load_enriched_block_frame(evaluation_dir)
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

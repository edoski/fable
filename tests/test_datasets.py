import unittest

import numpy as np

from spice_temporal.config import SplitConfig
from spice_temporal.constants import EVALUATION_END_TS, EVALUATION_START_TS
from spice_temporal.datasets import (
    action_count_for_delay,
    build_temporal_store,
    chronological_split_indices,
    derive_dataset_geometry,
    filter_sample_indices_by_anchor_window,
    history_context_blocks,
    lookback_steps_for_seconds,
    max_extra_wait_steps_for_delay,
    trim_history_blocks_for_target,
)
from spice_temporal.features import FEATURE_NAMES, build_feature_table
from spice_temporal.normalization import fit_standard_scaler
from spice_temporal.records import BlockRecord
from spice_temporal.torch_datasets import SequenceDataset


def make_block(index: int, base_fee: int, *, timestamp: int | None = None) -> BlockRecord:
    return BlockRecord(
        block_number=index,
        timestamp=timestamp if timestamp is not None else 1_700_000_000 + 12 * index,
        base_fee_per_gas=base_fee,
        gas_used=15_000_000 + index,
        gas_limit=30_000_000,
        chain_id=1,
    )


class DatasetLogicTestCase(unittest.TestCase):
    def test_fixed_step_mappings(self) -> None:
        self.assertEqual(lookback_steps_for_seconds(600, 12.0), 50)
        self.assertEqual(max_extra_wait_steps_for_delay(12, 12.0), 1)
        self.assertEqual(action_count_for_delay(12, 12.0), 2)
        self.assertEqual(max_extra_wait_steps_for_delay(24, 12.0), 2)
        self.assertEqual(action_count_for_delay(24, 12.0), 3)
        self.assertEqual(max_extra_wait_steps_for_delay(36, 12.0), 3)
        self.assertEqual(action_count_for_delay(36, 12.0), 4)
        self.assertEqual(max_extra_wait_steps_for_delay(36, 2.0), 18)
        self.assertEqual(action_count_for_delay(36, 2.0), 19)
        self.assertEqual(max_extra_wait_steps_for_delay(36, 1.6), 22)
        self.assertEqual(action_count_for_delay(36, 1.6), 23)

    def test_trim_history_blocks_for_target_uses_exact_tail_length(self) -> None:
        blocks = [make_block(index, 100 + index) for index in range(500)]
        geometry = derive_dataset_geometry(
            lookback_seconds=600,
            max_delay_seconds=36,
            block_time_seconds=12.0,
        )
        trimmed = trim_history_blocks_for_target(
            blocks,
            target_anchor_count=64,
            geometry=geometry,
        )
        self.assertEqual(len(trimmed), geometry.required_training_block_count(64))
        self.assertEqual(trimmed[0].block_number, 184)
        self.assertEqual(trimmed[-1].block_number, 499)

    def test_history_context_blocks_returns_exact_context_tail(self) -> None:
        blocks = [make_block(index, 100 + index) for index in range(500)]
        geometry = derive_dataset_geometry(
            lookback_seconds=600,
            max_delay_seconds=36,
            block_time_seconds=12.0,
        )
        context = history_context_blocks(blocks, geometry=geometry)
        self.assertEqual(len(context), geometry.context_block_count)
        self.assertEqual(context[0].block_number, 252)
        self.assertEqual(context[-1].block_number, 499)

    def test_build_temporal_store(self) -> None:
        blocks = [make_block(index, 100 + (index % 5)) for index in range(260)]
        table = build_feature_table(blocks)
        self.assertEqual(table.feature_matrix.shape[1], len(FEATURE_NAMES))
        store = build_temporal_store(table, lookback_steps=5, action_count=4)
        self.assertGreater(store.n_samples, 0)
        self.assertEqual(store.action_count, 4)
        self.assertEqual(store.class_labels.min(), 0)
        self.assertLess(store.class_labels.max(), 4)
        self.assertLessEqual(store.optimal_log_fee[0], store.next_block_log_fee[0])

    def test_filter_sample_indices_by_anchor_window(self) -> None:
        blocks = [
            make_block(
                index,
                100 + (index % 5),
                timestamp=EVALUATION_START_TS - 3_600 + 12 * index,
            )
            for index in range(1_000)
        ]
        table = build_feature_table(blocks)
        store = build_temporal_store(table, lookback_steps=50, action_count=4)
        filtered = filter_sample_indices_by_anchor_window(
            store,
            start_timestamp=EVALUATION_START_TS,
            end_timestamp=EVALUATION_END_TS,
        )
        self.assertGreater(filtered.shape[0], 0)
        anchor_timestamps = store.timestamps[store.anchor_row_indices[filtered]]
        self.assertGreaterEqual(int(anchor_timestamps[0]), EVALUATION_START_TS)
        self.assertLess(int(anchor_timestamps[-1]), EVALUATION_END_TS)

    def test_chronological_split_preserves_order(self) -> None:
        blocks = [make_block(index, 100 + index) for index in range(260)]
        store = build_temporal_store(build_feature_table(blocks), lookback_steps=5, action_count=4)
        split = chronological_split_indices(store.n_samples, SplitConfig())
        anchor_blocks = store.block_numbers[store.anchor_row_indices]
        self.assertLess(int(anchor_blocks[split.train[0]]), int(anchor_blocks[split.validation[0]]))
        self.assertLess(int(anchor_blocks[split.validation[0]]), int(anchor_blocks[split.test[0]]))

    def test_sequence_dataset_slices_expected_window(self) -> None:
        blocks = [make_block(index, 100 + (index % 5)) for index in range(260)]
        store = build_temporal_store(build_feature_table(blocks), lookback_steps=5, action_count=4)
        sample_indices = np.asarray([0], dtype=np.int64)
        dataset = SequenceDataset(store, sample_indices, lookback_steps=5)
        batch = dataset[0]
        expected = store.feature_matrix[:5]
        np.testing.assert_allclose(batch["inputs"].numpy(), expected)
        self.assertEqual(int(batch["class_label"].item()), int(store.class_labels[0]))

    def test_weighted_scaler_matches_naive_window_expansion(self) -> None:
        blocks = [make_block(index, 100 + (index % 7)) for index in range(320)]
        lookback_steps = 6
        store = build_temporal_store(
            build_feature_table(blocks),
            lookback_steps=lookback_steps,
            action_count=4,
        )
        split = chronological_split_indices(store.n_samples, SplitConfig())
        scaler = fit_standard_scaler(
            store.feature_matrix,
            anchor_row_indices=store.anchor_row_indices,
            sample_indices=split.train,
            lookback_steps=lookback_steps,
        )

        rows: list[np.ndarray] = []
        for sample_index in split.train.tolist():
            anchor_row_index = int(store.anchor_row_indices[sample_index])
            rows.append(
                store.feature_matrix[
                    anchor_row_index - lookback_steps + 1 : anchor_row_index + 1
                ]
            )
        flat = np.concatenate(rows, axis=0)
        np.testing.assert_allclose(np.asarray(scaler.means), flat.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(np.asarray(scaler.stds), flat.std(axis=0), atol=1e-5)


if __name__ == "__main__":
    unittest.main()

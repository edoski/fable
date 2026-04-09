import unittest

from spice_temporal.config import SplitConfig
from spice_temporal.datasets import (
    average_interblock_seconds,
    build_supervised_examples,
    chronological_split,
    earliest_min_offset,
    estimate_horizon_blocks,
)
from spice_temporal.features import build_feature_rows, feature_names
from spice_temporal.records import BlockRecord


def make_block(index: int, base_fee: int) -> BlockRecord:
    return BlockRecord(
        block_number=index,
        timestamp=1_700_000_000 + 12 * index,
        base_fee_per_gas=base_fee,
        gas_used=15_000_000 + index,
        gas_limit=30_000_000,
        chain_id=1,
    )


class DatasetLogicTestCase(unittest.TestCase):
    def test_estimate_horizon_blocks(self) -> None:
        self.assertEqual(estimate_horizon_blocks(36, 12.0), 3)
        self.assertEqual(estimate_horizon_blocks(12, 1.6), 7)

    def test_average_interblock_seconds(self) -> None:
        self.assertAlmostEqual(average_interblock_seconds([0, 12, 24, 36]), 12.0)

    def test_earliest_min_offset_breaks_ties_early(self) -> None:
        self.assertEqual(earliest_min_offset([4.0, 2.0, 2.0, 3.0]), 1)

    def test_build_supervised_examples(self) -> None:
        blocks = [make_block(index, 100 + (index % 5)) for index in range(260)]
        rows = build_feature_rows(blocks)
        self.assertEqual(len(rows[0].features), len(feature_names()))
        examples = build_supervised_examples(rows, lookback_steps=5, horizon_blocks=3)
        self.assertGreater(len(examples), 0)
        first = examples[0]
        self.assertEqual(len(first.inputs), 5)
        self.assertIn(first.class_label, {0, 1, 2})
        self.assertEqual(len(first.future_log_fees), 3)
        self.assertLessEqual(first.optimal_log_fee, first.next_block_log_fee)

    def test_chronological_split_preserves_order(self) -> None:
        blocks = [make_block(index, 100 + index) for index in range(260)]
        rows = build_feature_rows(blocks)
        examples = build_supervised_examples(rows, lookback_steps=5, horizon_blocks=3)
        split = chronological_split(examples, SplitConfig())
        self.assertLess(split.train[0].anchor_block_number, split.validation[0].anchor_block_number)
        self.assertLess(split.validation[0].anchor_block_number, split.test[0].anchor_block_number)


if __name__ == "__main__":
    unittest.main()

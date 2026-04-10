import json
import unittest
from dataclasses import asdict
from pathlib import Path

from spice_temporal.config import ChainConfig, ModelConfig, SplitConfig, TrainingConfig
from spice_temporal.constants import EVALUATION_START_TS
from spice_temporal.datasets import derive_dataset_geometry, lookback_steps_for_seconds
from spice_temporal.pipeline import (
    prepare_inference_dataset,
    prepare_training_dataset,
    run_training,
)
from spice_temporal.specs import TrainingSpec
from tests.support import make_evaluation_block, make_history_block


class PipelineTestCase(unittest.TestCase):
    def test_derive_lookback_steps(self) -> None:
        self.assertEqual(lookback_steps_for_seconds(600, 12.0), 50)

    def test_prepare_training_dataset_enforces_target_anchor_count(self) -> None:
        blocks = [make_history_block(index) for index in range(420)]
        spec = TrainingSpec(
            chain=ChainConfig(
                name="ethereum",
                chain_id=1,
                block_time_seconds=12.0,
                history_days=70,
            ),
            model=ModelConfig(family="lstm"),
            max_delay_seconds=12,
            lookback_seconds=600,
            target_anchor_count=64,
            split=SplitConfig(),
            training=TrainingConfig(device="cpu"),
        )
        prepared = prepare_training_dataset(
            blocks,
            spec=spec,
        )
        self.assertEqual(prepared.n_blocks_available, 420)
        self.assertEqual(prepared.n_examples_total, 64)
        self.assertEqual(prepared.geometry.lookback_steps, 50)
        self.assertEqual(prepared.geometry.max_extra_wait_steps, 1)
        self.assertEqual(prepared.geometry.action_count, 2)
        self.assertGreater(prepared.split_indices.train.shape[0], 0)
        self.assertEqual(prepared.n_features, prepared.store.feature_matrix.shape[1])

    def test_prepare_inference_dataset_uses_history_context(self) -> None:
        history_blocks = [make_history_block(index) for index in range(420)]
        evaluation_blocks = [make_evaluation_block(index) for index in range(720)]
        spec = TrainingSpec(
            chain=ChainConfig(
                name="ethereum",
                chain_id=1,
                block_time_seconds=12.0,
                history_days=70,
            ),
            model=ModelConfig(family="lstm"),
            max_delay_seconds=36,
            lookback_seconds=600,
            target_anchor_count=64,
            split=SplitConfig(),
            training=TrainingConfig(device="cpu"),
        )
        training = prepare_training_dataset(
            history_blocks,
            spec=spec,
        )
        geometry = derive_dataset_geometry(
            lookback_seconds=600,
            max_delay_seconds=36,
            block_time_seconds=12.0,
        )
        prepared = prepare_inference_dataset(
            history_blocks,
            evaluation_blocks,
            geometry=geometry,
            scaler=training.scaler,
        )
        self.assertEqual(prepared.n_history_context_blocks, geometry.context_block_count)
        self.assertEqual(prepared.n_evaluation_blocks, 720)
        self.assertGreater(prepared.n_examples_total, 0)
        anchor_timestamps = prepared.store.timestamps[
            prepared.store.anchor_row_indices[prepared.sample_indices]
        ]
        self.assertGreaterEqual(int(anchor_timestamps[0]), EVALUATION_START_TS)

    def test_run_training_on_json(self) -> None:
        blocks = [asdict(make_history_block(index)) for index in range(420)]
        fixture = Path("tests/fixtures_blocks.json")
        fixture.write_text(json.dumps(blocks), encoding="utf-8")
        try:
            result = run_training(
                history_block_path=fixture,
                spec=TrainingSpec(
                    chain=ChainConfig(
                        name="ethereum",
                        chain_id=1,
                        block_time_seconds=12.0,
                        history_days=70,
                    ),
                    model=ModelConfig(family="lstm"),
                    max_delay_seconds=12,
                    lookback_seconds=600,
                    target_anchor_count=64,
                    split=SplitConfig(),
                    training=TrainingConfig(
                        max_epochs=2,
                        effective_batch_size=8,
                        device="cpu",
                    ),
                ),
            )
        finally:
            fixture.unlink(missing_ok=True)
        self.assertEqual(result.prepared.n_blocks_available, 420)
        self.assertEqual(result.prepared.n_examples_total, 64)
        self.assertGreaterEqual(result.training_result.best_epoch, 0)
        self.assertGreaterEqual(result.test_metrics.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()

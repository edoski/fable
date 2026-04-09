import unittest
from dataclasses import asdict
from pathlib import Path

from spice_temporal.config import ChainConfig, ModelConfig, SplitConfig, TrainingConfig
from spice_temporal.pipeline import derive_lookback_steps, prepare_dataset, run_single_training
from spice_temporal.records import BlockRecord


def make_block(index: int) -> BlockRecord:
    return BlockRecord(
        block_number=index,
        timestamp=1_700_000_000 + 12 * index,
        base_fee_per_gas=100 + ((index // 3) % 7),
        gas_used=15_000_000 + (index % 1000),
        gas_limit=30_000_000,
        chain_id=1,
    )


class PipelineTestCase(unittest.TestCase):
    def test_derive_lookback_steps(self) -> None:
        self.assertEqual(derive_lookback_steps(600, 12.0), 50)

    def test_prepare_dataset(self) -> None:
        blocks = [make_block(index) for index in range(420)]
        prepared = prepare_dataset(
            blocks,
            chain=ChainConfig(
                name="ethereum",
                chain_id=1,
                nominal_block_time_seconds=12.0,
                history_days_hint=70,
                rpc_env_var="ETH_RPC_URL",
            ),
            window_seconds=12,
            lookback_seconds=600,
            split_config=SplitConfig(),
        )
        self.assertEqual(prepared.lookback_steps, 50)
        self.assertGreater(prepared.horizon_blocks, 0)
        self.assertGreater(len(prepared.train_examples), 0)
        self.assertEqual(prepared.n_features, len(prepared.train_examples[0].inputs[0]))

    def test_run_single_training_on_json(self) -> None:
        blocks = [asdict(make_block(index)) for index in range(420)]
        fixture = Path("tests/fixtures_blocks.json")
        fixture.write_text(__import__("json").dumps(blocks), encoding="utf-8")
        try:
            result = run_single_training(
                block_file=fixture,
                chain=ChainConfig(
                    name="ethereum",
                    chain_id=1,
                    nominal_block_time_seconds=12.0,
                    history_days_hint=70,
                    rpc_env_var="ETH_RPC_URL",
                ),
                window_seconds=12,
                lookback_seconds=600,
                model_config=ModelConfig(family="lstm"),
                training_config=TrainingConfig(max_epochs=2, effective_batch_size=8, device="cpu"),
                split_config=SplitConfig(),
            )
        finally:
            fixture.unlink(missing_ok=True)
        self.assertGreaterEqual(result.training_result.best_epoch, 0)
        self.assertGreaterEqual(result.test_metrics.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()

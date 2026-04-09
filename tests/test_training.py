import unittest

import torch

from spice_temporal.config import ModelConfig, TrainingConfig
from spice_temporal.evaluation import BatchMetrics
from spice_temporal.models import build_model
from spice_temporal.normalization import fit_standard_scaler, transform_examples
from spice_temporal.records import SupervisedExample
from spice_temporal.training import _mean_metrics, train_model


def make_example(anchor: int, label: int) -> SupervisedExample:
    future = [4.0, 3.0, 2.0]
    if label == 0:
        future = [1.0, 2.0, 3.0]
    elif label == 1:
        future = [3.0, 1.0, 2.0]
    return SupervisedExample(
        anchor_block_number=anchor,
        anchor_timestamp=1_700_000_000 + anchor,
        inputs=[[float(label), float(step)] for step in range(5)],
        class_label=label,
        target_log_fee=future[label],
        future_log_fees=future,
        next_block_log_fee=future[0],
        optimal_log_fee=min(future),
    )


class TrainingSmokeTestCase(unittest.TestCase):
    def test_epoch_metrics_are_weighted_by_batch_size(self) -> None:
        summary = _mean_metrics(
            [
                BatchMetrics(
                    count=4,
                    total_loss=1.0,
                    accuracy=0.5,
                    mean_cost_over_optimum=0.2,
                    mean_profit_over_baseline=0.1,
                ),
                BatchMetrics(
                    count=1,
                    total_loss=3.0,
                    accuracy=1.0,
                    mean_cost_over_optimum=0.8,
                    mean_profit_over_baseline=0.4,
                ),
            ]
        )
        self.assertAlmostEqual(summary.total_loss, 1.4)
        self.assertAlmostEqual(summary.accuracy, 0.6)
        self.assertAlmostEqual(summary.mean_cost_over_optimum, 0.32)
        self.assertAlmostEqual(summary.mean_profit_over_baseline, 0.16)

    def test_lstm_smoke_train(self) -> None:
        train_examples = [make_example(index, index % 3) for index in range(24)]
        validation_examples = [make_example(100 + index, index % 3) for index in range(6)]
        scaler = fit_standard_scaler(train_examples)
        train_examples = transform_examples(train_examples, scaler)
        validation_examples = transform_examples(validation_examples, scaler)
        model = build_model(
            n_features=2,
            n_classes=3,
            config=ModelConfig(family="lstm"),
        )
        result = train_model(
            model,
            train_examples=train_examples,
            validation_examples=validation_examples,
            config=TrainingConfig(max_epochs=2, effective_batch_size=8, device="cpu"),
        )
        self.assertGreaterEqual(result.best_epoch, 0)
        self.assertGreater(len(result.train_history), 0)
        with torch.no_grad():
            inputs = torch.tensor(train_examples[0].inputs, dtype=torch.float32).unsqueeze(0)
            outputs = model(inputs)
        self.assertEqual(outputs.logits.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()

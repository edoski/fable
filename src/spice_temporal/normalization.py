"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

from spice_temporal.records import SupervisedExample


@dataclass(slots=True)
class StandardScaler:
    means: list[float]
    stds: list[float]

    def transform_vector(self, vector: list[float]) -> list[float]:
        return [
            (value - mean) / std if std > 0.0 else value - mean
            for value, mean, std in zip(vector, self.means, self.stds, strict=True)
        ]


def fit_standard_scaler(examples: list[SupervisedExample]) -> StandardScaler:
    if not examples:
        raise ValueError("Need at least one example to fit a scaler")
    n_features = len(examples[0].inputs[0])
    totals = [0.0] * n_features
    totals_sq = [0.0] * n_features
    count = 0
    for example in examples:
        for row in example.inputs:
            for index, value in enumerate(row):
                totals[index] += value
                totals_sq[index] += value * value
            count += 1

    means = [value / count for value in totals]
    stds: list[float] = []
    for index in range(n_features):
        mean = means[index]
        variance = max(totals_sq[index] / count - mean * mean, 0.0)
        stds.append(variance**0.5)
    return StandardScaler(means=means, stds=stds)


def transform_examples(
    examples: list[SupervisedExample],
    scaler: StandardScaler,
) -> list[SupervisedExample]:
    transformed: list[SupervisedExample] = []
    for example in examples:
        transformed.append(
            SupervisedExample(
                anchor_block_number=example.anchor_block_number,
                anchor_timestamp=example.anchor_timestamp,
                inputs=[scaler.transform_vector(row) for row in example.inputs],
                class_label=example.class_label,
                target_log_fee=example.target_log_fee,
                future_log_fees=list(example.future_log_fees),
                next_block_log_fee=example.next_block_log_fee,
                optimal_log_fee=example.optimal_log_fee,
            )
        )
    return transformed

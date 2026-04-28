"""Shared offset-classification metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OffsetClassificationCounts:
    correct_count: int
    true_positive_by_class: tuple[int, ...]
    predicted_by_class: tuple[int, ...]
    target_by_class: tuple[int, ...]


def offset_classification_counts(
    predicted_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    *,
    n_classes: int,
) -> OffsetClassificationCounts:
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    predicted = predicted_offsets.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    target = target_offsets.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("predicted_offsets and target_offsets must have matching shape")
    if predicted.numel() == 0:
        raise ValueError("offset classification metrics require at least one sample")
    if bool(((predicted < 0) | (predicted >= n_classes)).any()):
        raise ValueError("predicted_offsets contain values outside the action width")
    if bool(((target < 0) | (target >= n_classes)).any()):
        raise ValueError("target_offsets contain values outside the action width")

    correct_mask = predicted == target
    true_positive = torch.bincount(
        target[correct_mask],
        minlength=n_classes,
    )
    predicted_count = torch.bincount(predicted, minlength=n_classes)
    target_count = torch.bincount(target, minlength=n_classes)
    return OffsetClassificationCounts(
        correct_count=int(correct_mask.sum().item()),
        true_positive_by_class=_count_tuple(true_positive, n_classes=n_classes),
        predicted_by_class=_count_tuple(predicted_count, n_classes=n_classes),
        target_by_class=_count_tuple(target_count, n_classes=n_classes),
    )


def add_offset_classification_counts(
    left: OffsetClassificationCounts | None,
    right: OffsetClassificationCounts,
) -> OffsetClassificationCounts:
    if left is None:
        return right
    if len(left.true_positive_by_class) != len(right.true_positive_by_class):
        raise ValueError("offset classification count widths do not match")
    return OffsetClassificationCounts(
        correct_count=left.correct_count + right.correct_count,
        true_positive_by_class=_add_count_tuples(
            left.true_positive_by_class,
            right.true_positive_by_class,
        ),
        predicted_by_class=_add_count_tuples(left.predicted_by_class, right.predicted_by_class),
        target_by_class=_add_count_tuples(left.target_by_class, right.target_by_class),
    )


def macro_f1_from_counts(counts: OffsetClassificationCounts) -> float:
    values: list[float] = []
    for true_positive, predicted_count, target_count in zip(
        counts.true_positive_by_class,
        counts.predicted_by_class,
        counts.target_by_class,
        strict=True,
    ):
        if target_count <= 0:
            continue
        precision = true_positive / predicted_count if predicted_count > 0 else 0.0
        recall = true_positive / target_count
        if precision <= 0.0 or recall <= 0.0:
            values.append(0.0)
            continue
        values.append(2.0 * precision * recall / (precision + recall))
    if not values:
        raise ValueError("macro_f1 requires at least one supported target class")
    return sum(values) / len(values)


def _count_tuple(values: torch.Tensor, *, n_classes: int) -> tuple[int, ...]:
    return tuple(int(value) for value in values[:n_classes].tolist())


def _add_count_tuples(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )

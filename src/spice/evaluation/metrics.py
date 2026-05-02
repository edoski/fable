"""Evaluator metric descriptors."""

from __future__ import annotations

from ..metrics import MetricDescriptor

TEMPORAL_REPLAY_METRIC_DESCRIPTORS: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        id="profit_over_baseline",
        label="profit over baseline",
        role="primary",
    ),
    MetricDescriptor(
        id="cost_over_optimum",
        label="cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="baseline_cost_over_optimum",
        label="baseline cost over optimum",
        role="secondary",
    ),
    MetricDescriptor(
        id="exact_optimum_hit_rate",
        label="exact optimum hit rate",
        role="secondary",
    ),
    MetricDescriptor(
        id="realized_fee_sum",
        label="realized fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="baseline_fee_sum",
        label="baseline fee sum",
        role="diagnostic",
    ),
    MetricDescriptor(
        id="optimum_fee_sum",
        label="optimum fee sum",
        role="diagnostic",
    ),
)

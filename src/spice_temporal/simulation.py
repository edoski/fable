"""Economic simulation helpers for temporal evaluation."""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

from spice_temporal.records import SupervisedExample


@dataclass(slots=True)
class SimulationSummary:
    mean_profit_over_baseline: float
    mean_cost_over_optimum: float
    n_events: int


def sample_poisson_arrivals(
    *,
    rate_per_second: float,
    start_timestamp: int,
    end_timestamp: int,
    seed: int,
) -> list[float]:
    if rate_per_second <= 0:
        raise ValueError("rate_per_second must be positive")
    random.seed(seed)
    arrivals: list[float] = []
    cursor = float(start_timestamp)
    while cursor < end_timestamp:
        cursor += random.expovariate(rate_per_second)
        if cursor < end_timestamp:
            arrivals.append(cursor)
    return arrivals


def summarize_realized_costs(
    examples: list[SupervisedExample],
    predicted_offsets: list[int],
) -> SimulationSummary:
    if len(examples) != len(predicted_offsets):
        raise ValueError("examples and predicted_offsets must have the same length")
    profits: list[float] = []
    costs: list[float] = []
    for example, predicted_offset in zip(examples, predicted_offsets, strict=True):
        realized = math.exp(example.future_log_fees[predicted_offset])
        baseline = math.exp(example.next_block_log_fee)
        optimum = math.exp(example.optimal_log_fee)
        profits.append((baseline - realized) / baseline)
        costs.append((realized - optimum) / optimum)
    return SimulationSummary(
        mean_profit_over_baseline=sum(profits) / len(profits),
        mean_cost_over_optimum=sum(costs) / len(costs),
        n_events=len(profits),
    )


def select_examples_for_arrivals(
    examples: list[SupervisedExample],
    arrivals: list[float],
) -> list[SupervisedExample]:
    timestamps = [example.anchor_timestamp for example in examples]
    selected: list[SupervisedExample] = []
    for arrival in arrivals:
        index = bisect.bisect_right(timestamps, arrival) - 1
        if index >= 0:
            selected.append(examples[index])
    return selected

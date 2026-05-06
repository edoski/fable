# Concrete Evaluator

Evaluators score decoded model decisions against historical fee data. They are separate from prediction losses: a model can train with one loss and still be evaluated with replay-style economic metrics.

## Mental Model

Evaluation starts after inference:

```text
model outputs
  -> prediction decode
  -> DecodedOffsets
  -> evaluator contract
  -> EvaluationSummary
```

Current evaluators accept `DecodedOffsets`. They interpret offsets through temporal rows and the execution policy.

## Evaluation Contract

`CompiledEvaluatorContract` contains:

| Field | Meaning |
| --- | --- |
| `evaluator_id` | Stable evaluator config id. |
| `metric_descriptors` | Metric names, roles, and directions. |
| `accepted_decoded_result_id` | Decoded ABI the evaluator accepts. |
| `run_fn` | Concrete evaluation function. |

The contract validates metric descriptor uniqueness, requires exactly one descriptor with `role: primary`, and validates decoded-result id before running. `primary_metric_id` is derived from that primary descriptor; optimization direction belongs to metric descriptors.

## Fee Math

Problem stores keep fees in log space. Evaluators exponentiate before economic ratios:

```text
fee = exp(log_base_fee)
```

Ratios are computed on real fee values, not log values. The Temporal Replay Runner validates decoded replay inputs and asks evaluator Adapters for selected events; Temporal Accounting is shared after that selection.

Temporal replay has a private typed result ABI between Temporal Accounting and the Temporal Replay Runner. It carries run metrics, event metric sums, window summaries, and metadata as replay concepts. Aggregate ratio metrics are event-weighted across all replay events; `window_metrics` summarize per-run ratio means without event weighting. The runner validates metadata as strings, integers, or floats only, then converts the replay result to generic `EvaluationSummary` at the evaluator boundary.

## Temporal Accounting

Temporal Accounting receives selected sample positions from the Temporal Replay Runner and computes realized, baseline, and optimum fee outcomes.

```text
selected positions
  -> execution policy
  -> realized rows
  -> fee ratios and sums
```

Shared metrics include:

| Metric | Formula idea |
| --- | --- |
| `profit_over_baseline` | Mean `(baseline - realized) / baseline` per event. |
| `cost_over_optimum` | Mean `(realized - optimum) / optimum` per event. |
| `baseline_cost_over_optimum` | Mean `(baseline - optimum) / optimum` per event. |
| `exact_optimum_hit_rate` | Fraction of events whose realized row equals the optimum row. |
| Fee sums | Realized, baseline, optimum totals. |

## Poisson Replay

`poisson_replay_2h` simulates randomly timed transaction opportunities. Its adapter owns replay windows, exponential inter-arrival sampling, and arrival-to-sample mapping. Each repetition picks a uniform replay window, maps each arrival to the latest sample at or before that timestamp, then uses Temporal Accounting.

```text
arrivals
  -> sample positions
  -> Temporal Accounting
```

Duplicate selected samples can occur when several arrivals map to the same sample. They count as repeated events and contribute independently to event-weighted aggregate metrics.

## Full Temporal Replay

`full_temporal_replay` selects every supplied sample position exactly once, then uses Temporal Accounting. In train and tune objectives, supplied samples are validation samples. In the evaluate workflow, supplied samples are held-out evaluation-window samples.

## Objective Link

Evaluation objectives call evaluator scoring on validation samples during training and tuning. Standalone evaluate workflow compiles the selected evaluation config and stores its summary under the artifact state DB.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Unknown evaluator id | Evaluator id is not one of the trusted adapters. |
| Decoded-result id mismatch | Evaluator cannot interpret prediction output. |
| Empty selected positions | No events to score. |
| Non-positive fee total | Economic ratio denominator invalid. |
| Poisson window too long | Not enough timestamp coverage. |
| Multiple stored summaries without id | Artifact state has several evaluation summaries. |

## Extension Pattern

A future evaluator should declare its accepted decoded-result id and metric descriptors, with exactly one primary descriptor. Reuse Temporal Accounting when the evaluator scores selected temporal decisions.

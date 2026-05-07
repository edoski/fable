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

The contract validates metric descriptor uniqueness, requires exactly one descriptor with `role: primary`, validates decoded-result id before running, and validates returned summary/run metric ids against descriptors. `primary_metric_id` is derived from that primary descriptor; optimization direction belongs to metric descriptors.

## Fee Math

Problem stores keep fees in log space. Evaluators exponentiate before economic ratios:

```text
fee = exp(log_base_fee)
```

Ratios are computed on real fee values, not log values. The Temporal Replay Runner validates decoded replay inputs, builds the replay sample view from supplied sample positions, timestamps, and count, asks evaluator Adapters for selected events, normalizes scalar metadata, handles no-run failures, invokes Temporal Accounting, and converts replay results to `EvaluationSummary`.

Temporal replay has a private typed result ABI between Temporal Accounting and the Temporal Replay Runner. It carries run metrics, fee sums, event metric sums, window summaries, and metadata as replay concepts. Fee sums remain accounting facts until metric assembly instead of being recovered from generic metric dictionaries. The Temporal Replay Metric Catalog owns metric ids, descriptors, event-mean membership, fee-sum membership, window-summary membership, metric assembly, and extraction to generic metric dictionaries. Event-mean metrics are event-weighted across all replay events; `window_metrics` summarize per-run means without event weighting. The runner converts the replay result to generic `EvaluationSummary` at the evaluator boundary.

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

`poisson_replay` simulates randomly timed transaction opportunities. Its adapter owns replay windows, exponential inter-arrival sampling, chronological ordering of the replay sample view, and arrival-to-sample mapping. Each repetition picks a uniform replay window and maps each arrival to the latest sample at or before that timestamp. The Temporal Replay Runner accounts those selected positions.

```text
arrivals
  -> sample positions
  -> Temporal Accounting
```

Duplicate selected samples can occur when several arrivals map to the same sample. They count as repeated events and contribute independently to event-weighted aggregate metrics.

## Full Temporal Replay

`full_temporal_replay` selects every supplied sample position exactly once, then the Temporal Replay Runner accounts those selected positions. In train and tune objectives, supplied samples are validation samples. In the evaluate workflow, supplied samples are held-out evaluation-window samples.

## Objective Link

Evaluation objectives call evaluator scoring on validation samples during training and tuning. Standalone evaluate workflow compiles the selected evaluation config and stores its summary under the artifact state DB.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Unknown evaluator id | Evaluator id is not one of the trusted adapters. |
| Decoded-result id mismatch | Evaluator cannot interpret prediction output. |
| Evaluator metric id mismatch | Returned summary/run metrics do not match descriptor ids, or a window metric is undeclared. |
| Empty selected positions | No events to score. |
| Non-positive fee total | Economic ratio denominator invalid. |
| Poisson window too long | Not enough timestamp coverage. |
| Multiple stored summaries without id | Artifact state has several evaluation summaries. |

## Extension Pattern

A future temporal replay evaluator should provide only event selection and selection provenance to the Temporal Replay Runner. Add a separate evaluator path only when scoring semantics are not temporal-decision replay.

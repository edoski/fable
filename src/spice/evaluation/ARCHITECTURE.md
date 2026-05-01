# Evaluation Architecture

## Purpose

`evaluation` scores decoded predictions against temporal problem stores. It owns evaluator config validation, evaluator contracts, metric descriptors, event selection, shared Temporal Accounting, and decoded-result checks.

Training metrics answer “did the model optimize its objective?” Evaluation metrics answer “were the decoded decisions good under the temporal problem?” These are related but not identical.

## Config And Registry Pattern

There are two concrete evaluator specs:

```yaml
id: poisson_replay_2h
window_seconds: 7200
repetitions: 50
arrival_rate_per_second: 0.05
seed: 2026
```

```yaml
id: full_temporal_replay
```

```text
mapping / EvaluatorConfig
        |
        v
coerce_evaluator_config()
  - dispatch by evaluation.id
  - validate concrete evaluator config
        |
        v
compile_evaluator_contract()
  - dispatch by config id
  - compile concrete evaluator contract
```

The registry is an explicit table of trusted evaluator adapters. It is not a generic engine registry.

## Scoring Flow

```text
model outputs
    |
    v
prediction contract decodes result
    |
    v
decoded result id
    |
    +--> evaluator contract checks accepted decoded-result id
             |
             v
        evaluator.run(store, execution_policy, decoded_result, sample_indices)
             |
             v
        EvaluationSummary
```

Evaluators never call models or prediction heads. The modeling scoring service performs inference and passes decoded results to evaluator contracts.

## Decoded Result Boundary

The evaluator accepts the generic `DecodedPredictionResult` from `prediction.decoding` at the evaluator entrypoint because that is the evaluator ABI. Replay then narrows to `DecodedOffsets` from `prediction.decoded_offsets` once:

```text
Temporal Replay Runner
  generic decoded result -> require_decoded_offsets()
  DecodedOffsets -> temporal_accounting.summarize_selected_temporal_decisions()
```

`temporal_accounting` is evaluation-private and accepts `DecodedOffsets` directly. That keeps generic decoded-result handling at the evaluator boundary.

## Temporal Accounting

Temporal replay evaluator Adapters choose event positions. The **Temporal Replay Runner** owns decoded-result validation and the accounting loop. Temporal Accounting computes realized, baseline, optimum, and event-mean economic metrics for those positions.

```text
selected sample positions
  -> execution policy realization
  -> realized / baseline / optimum rows
  -> real fee values
  -> event-mean metrics and fee sums
```

`poisson_replay_2h` selects positions by sampled arrivals. `full_temporal_replay` selects every supplied sample position once. Both use the same Temporal Accounting implementation.

## Metric Descriptor Rule

```text
metric id -> label -> role(primary | secondary | diagnostic)
```

Each evaluator contract must declare exactly one primary metric. Descriptor ids must be unique and path-safe. The primary metric drives objective alignment and reporting; secondary and diagnostic metrics explain behavior.

## Module Map

```text
evaluation/
  config.py          evaluator config models
  contracts.py       compiled evaluator contract and summary models
  registry.py        public config coercion and contract compile helpers
  metrics.py         metric descriptor definitions
  sampling.py        Poisson arrival and chronological sample helpers
  temporal_replay_runner.py shared decoded validation and replay accounting loop
  poisson_replay.py  Poisson replay evaluator adapter
  full_temporal_replay.py full supplied-sample evaluator adapter
  temporal_accounting.py shared temporal decision accounting
  summary.py         generic summary helpers
```

## Extension Points

Add another evaluator adapter only when it has distinct event-selection or scoring semantics. Workflows should keep calling `compile_evaluator_contract()`.

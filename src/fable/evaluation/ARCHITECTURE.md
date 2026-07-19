# Evaluation Architecture

## Purpose

`evaluation` scores decoded predictions against temporal problem stores. It owns evaluator config validation, evaluator contracts, evaluator metric descriptors, event selection, shared Temporal Accounting, and decoded-result checks. Generic metric result types live in `spice.metrics`.

Training metrics answer “did the prediction loss improve?” Evaluation metrics answer “were the decoded decisions good under the temporal problem?” These are related but not identical.

## Config And Registry Pattern

There are two concrete evaluator specs:

```yaml
id: poisson_replay
window_seconds: 7200
repetitions: 50
arrival_rate_per_second: 0.05
seed: 2026
```

```yaml
```

```text
mapping / EvaluatorConfig
        |
        v
coerce_evaluator_config()
  - dispatch by evaluation.id
  - validate concrete evaluator config
  - report config-facing envelope errors as ConfigResolutionError
        |
        v
compile_evaluator_contract()
  - dispatch by config id
  - compile concrete evaluator contract
```

The registry is an explicit table of trusted evaluator adapters. It is not a generic engine registry. Shared owner-spec helpers handle only mechanical payload and concrete-type validation; evaluator ids and compile hooks stay local to `evaluation`.

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
        evaluator.run(store, execution_policy, decoded_result, action_space)
             |
             v
        EvaluationSummary
```

Evaluators never call models or prediction heads. The `modeling.scoring` bridge performs inference and passes decoded results to evaluator contracts.

## Decoded Result Boundary

The evaluator accepts the generic `DecodedPredictionResult` and a prepared Action Space at the evaluator entrypoint because that is the evaluator ABI. Replay then narrows to `DecodedOffsets` from `prediction.decoded_offsets` once:

```text
Temporal Replay Runner
  generic decoded result -> require_decoded_offsets()
  validated selected runs -> temporal_accounting.summarize_selected_temporal_decision_runs()
```

`temporal_accounting` is evaluation-private and accepts `DecodedOffsets` directly. That keeps generic decoded-result handling at the evaluator boundary.

## Temporal Accounting

Temporal replay evaluator Adapters choose event positions from a runner-built replay sample view. The **Temporal Replay Runner** owns decoded-result validation, replay sample positions/timestamps/count derived from the prepared Action Space, selected-position validation, strict scalar metadata validation, and no-run failures. Temporal Accounting computes realized, baseline, optimum, event-mean economic metrics, fee sums, and per-run window summaries for validated selected runs.

```text
selected sample positions
  -> execution policy realization
  -> realized / baseline / optimum rows
  -> real fee values
  -> event-mean metrics and fee sums
```

`poisson_replay` owns Poisson windowing, arrival sampling, chronological ordering of the replay sample view, and arrival-to-position selection. It feeds selections to the runner, which validates metadata, handles no-run failures, and uses Temporal Accounting.

Temporal Accounting returns evaluation-private temporal replay result types. Event metric sums and fee sums stay as replay accounting facts until output assembly. A Temporal Replay Metric Catalog owns metric ids, descriptors, event-mean membership, fee-sum membership, window-summary membership, metric validation, metric assembly, and extraction into generic metric dictionaries. The Temporal Replay Runner attaches those descriptors and converts typed replay results to generic `EvaluationSummary` before the result leaves `evaluation`, so storage, benchmarks, reporting, and modeling keep one public evaluation result ABI.

## Metric Descriptor Rule

```text
metric id -> label -> role(primary | secondary | diagnostic)
```

Each evaluator contract must declare exactly one primary metric. Descriptor ids must be unique and path-safe. Returned summary metrics and per-run metrics must exactly match descriptor ids; window metric ids must be descriptor ids. Metric descriptors may also declare optimization direction for reporting and comparison.

## Module Map

```text
evaluation/
  config.py          evaluator config models
  contracts.py       compiled evaluator contract and summary models
  registry.py        public config coercion and contract compile helpers
  temporal_replay_runner.py decoded validation, replay loop, summary conversion
  _temporal_replay_metric_catalog.py private metric ids, descriptors, validation, aggregation facts, assembly, and extraction
  temporal_replay_results.py private typed temporal replay results
  poisson_replay.py  Poisson replay evaluator adapter and event selection policy
  temporal_accounting.py temporal decision accounting math
```

## Extension Points

Add another evaluator adapter only when it has distinct event-selection or scoring semantics. Workflows should keep calling `compile_evaluator_contract()`.

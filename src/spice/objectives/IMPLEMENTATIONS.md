# Concrete Objectives

Objectives define what training and tuning optimize. They are not model losses by themselves; they select a metric and a direction.

## Mental Model

Training always computes prediction metrics. An objective decides which metric controls best-state selection and early stopping.

```text
validation batches
  -> prediction metrics
  -> Objective Runtime
  -> objective metric
  -> best-state decision
```

## Validation Objective

`id: validation` uses validation metrics directly. Current checked-in spec `validation_total_loss` minimizes `total_loss`.

This is the simplest objective: it optimizes the same metric produced by the prediction family during validation.

## Evaluation Objective

`id: evaluation` selects an evaluator metric from validation-sample scoring. `modeling.objective_runtime` runs the model-bound scoring path. Current checked-in specs maximize `profit_over_baseline` under Poisson replay or full temporal replay evaluators.

```text
validation samples
  -> predict offsets
  -> run evaluator
  -> select evaluator metric
```

This is slower than a validation-loss objective, but it aligns best-state selection with the economic metric used for research reporting.

## Direction

Each objective has a direction:

| Direction | Meaning |
| --- | --- |
| `minimize` | Smaller metric is better. |
| `maximize` | Larger metric is better. |

The direction controls early stopping, best-state selection, and Optuna study optimization. For evaluation objectives, the direction must match the selected evaluator metric descriptor when that descriptor declares one.

## Evaluator Binding

Evaluation objectives name an evaluator id. Train and tune configs must select the same evaluation, and the resolved evaluator config is part of study/artifact identity. This keeps the optimized metric tied to the intended evaluator and its concrete replay settings.

Evaluate workflow can run a selected diagnostic evaluator directly; artifact inference validates manifest and selected corpus compatibility.

## Current Specs

| Spec | Mode | Metric | Direction |
| --- | --- | --- | --- |
| `validation_total_loss` | validation | `total_loss` | minimize |
| `profit_poisson_replay` | evaluation | `profit_over_baseline` | maximize |
| `profit_full_temporal_replay` | evaluation | `profit_over_baseline` | maximize |

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Metric id missing | Objective cannot find configured metric. |
| Evaluation objective without evaluator | Evaluator scoring target is undefined. |
| Evaluator mismatch | Training would optimize a different evaluator than requested. |
| Direction mismatch | Objective direction contradicts the evaluator metric descriptor. |
| Invalid direction | Best-metric comparison is undefined. |

## Extension Pattern

A new objective mode should still produce one metric id and direction. Keep objective selection separate from prediction family loss implementation.

# Workflows Architecture

## Purpose

`workflows` orchestrates application tasks: acquire, train, tune, and evaluate. A workflow connects generic services in the correct order, validates preconditions, reports progress, and commits outputs.

It should not own implementation details from feature families, prediction families, model families, evaluator adapters, or storage internals.

## Boundary Rule

```text
resolved typed config
        |
        v
workflow preparation
        |
        v
workflow runner
        |
        +--> contracts and services
        +--> storage transactions
        +--> reporter messages
```

Workflows receive typed configs. They do not resolve remote target defaults. They do not branch on concrete evaluator or prediction implementation ids.

`preparation.py` is the generic preflight seam. It resolves consumed/produced roots, loads root manifests, applies tuned training params, builds workflow specs, and validates coverage. Runner modules then report progress, call owner packages, and perform one explicit storage effect.

## Acquire

```text
AcquireConfig
  -> prepare produced corpus root
  -> report acquisition plan
  -> create block source with corpus-derived source requirements
  -> delegate corpus assembly
  -> close block source
```

Acquire is deliberately thin. Corpus Assembly owns capability planning, source requirements, bounded refill attempts, split materialization, state writing, and commit/reindex mechanics.

## Train

```text
TrainConfig
  -> prepare train roots, manifest, active config, and training spec
  -> open full-root transaction
  -> persisted training
  -> promote artifact root
  -> report result
```

Train uses complete-root staging because it produces a full artifact root.

## Tune

```text
TuneConfig
  -> prepare tune roots, manifest, and coverage spec
  -> delegate study opening and trial execution to modeling.tuning_execution
  -> reindex study root after materialization
```

Tune mutates study state rather than staging an entire root for each trial.

## Evaluate

```text
EvaluateConfig
  -> prepare evaluation roots and inference context
  -> score model with evaluator through modeling.scoring
  -> upsert evaluation state with execution provenance when remote
  -> report result
```

Evaluate appends or updates artifact state. It does not stage a full artifact root.

## Extension Points

Add workflow code only for task-level orchestration. If logic is reusable and domain-specific, put it in the owning package first, then call it from the workflow.

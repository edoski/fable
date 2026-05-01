# Workflows Architecture

## Purpose

`workflows` orchestrates application tasks: acquire, train, tune, and evaluate. A workflow connects generic services in the correct order, validates preconditions, reports progress, and commits outputs.

It should not own implementation details from feature families, prediction families, model families, evaluator engines, or storage internals.

## Boundary Rule

```text
resolved typed config
        |
        v
workflow
        |
        +--> contracts and services
        +--> storage layout/staging primitives
        +--> reporter messages
```

Workflows receive typed configs. They do not resolve remote target defaults. They do not branch on concrete evaluator or prediction implementation ids.

## Acquire

```text
AcquireConfig
  -> derive produced corpus root
  -> report acquisition plan
  -> create block source
  -> delegate corpus assembly
  -> close block source
```

Acquire is deliberately thin. Corpus Assembly owns capability planning, bounded refill attempts, split materialization, state writing, and commit/reindex mechanics.

## Train

```text
TrainConfig
  -> resolve consumed roots through catalog
  -> for tuned train, apply best tuned params
  -> load resolved corpus manifest
  -> derive produced artifact root
  -> build artifact training spec
  -> validate corpus coverage
  -> staged artifact root
  -> persisted training
  -> validate/promote artifact root
  -> report result
```

Train uses complete-root staging because it produces a full artifact root.

## Tune

```text
TuneConfig
  -> resolve consumed dataset root through catalog
  -> derive produced study root
  -> build trial training specs
  -> validate corpus coverage
  -> materialize or update study state
  -> run trials
  -> reindex study root after materialization
```

Tune mutates study state rather than staging an entire root for each trial.

## Evaluate

```text
EvaluateConfig
  -> resolve dataset and artifact roots
  -> load artifact
  -> prepare inference dataset
  -> score model with evaluator through modeling.scoring
  -> upsert evaluation state with execution provenance when remote
  -> report result
```

Evaluate appends or updates artifact state. It does not stage a full artifact root.

## Extension Points

Add workflow code only for task-level orchestration. If logic is reusable and domain-specific, put it in the owning package first, then call it from the workflow.

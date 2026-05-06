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
        +--> explicit storage-owned effect boundaries
        +--> reporter messages
```

Workflows receive typed configs. They do not resolve remote target defaults. They do not branch on concrete evaluator or prediction implementation ids.

`preparation.py` is the generic preflight seam. It calls Storage Root Materialization for root handles, then keeps tuned active config, training/tuning coverage preflight, and inference context preparation in focused workflow-owned modules. Runner modules then report progress, call owner packages, and cross explicit storage-owned effect boundaries.

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
  -> storage full-root commit runs persisted training in a staged root
  -> report result
```

Train uses complete-root staging because it produces a full artifact root. The workflow supplies the training writer; storage owns promotion, cleanup, root-kind validation, and catalog reindex.

## Tune

```text
TuneConfig
  -> prepare tune roots, manifest, and coverage spec
  -> delegate study opening and trial execution to modeling.tuning_execution
  -> storage records study-root open/reindex and trial mutation/reindex effects
```

Tune mutates study state rather than staging an entire root for each trial.

## Evaluate

```text
EvaluateConfig
  -> prepare evaluation roots and inference context
  -> score model with evaluator through modeling.scoring
  -> storage records evaluation state with execution provenance when remote
  -> report result
```

Evaluate appends or updates artifact state. It does not stage a full artifact root, and storage intentionally does not reindex the artifact catalog for evaluation summaries.

## Extension Points

Add workflow code only for task-level orchestration. If logic is reusable and domain-specific, put it in the owning package first, then call it from the workflow.

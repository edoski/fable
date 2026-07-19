# Temporal Compilers Architecture

## Purpose

`temporal.compilers` owns concrete ways to lower feature tables into problem stores.

## Theory

A temporal compiler is a temporal geometry engine. It answers: for each decision row, how much history is visible, where the candidate window starts and ends, and which rows are available for execution-policy outcome facts. It does not own action availability, prediction targets, or evaluator metrics.

## Pattern

Compiler config validates user intent. The local registry maps compiler id to config types, compile hooks, and runtime metadata codecs. `core.specs` supplies the mechanical owner-spec helper for payload coercion; compiler ids and runtime metadata codecs stay local. The compiled problem contract then builds stores for training and delay-specific stores for evaluation.

Config-facing compiler and observed-time-window slot-spacing payload errors normalize to `ConfigResolutionError`. Runtime metadata codecs keep their runtime metadata error domain.

## Invariants

Compilers must publish feature prerequisites. They must serialize runtime metadata through compiler-owned codecs. Dataset builders and workflows must call compiler contracts rather than inspect concrete compiler types.

## Extension Points

Add a compiler when example construction changes. Do not add compiler branches in workflows; add one registry entry and a contract implementation.

## Compiler Flow

```text
ProblemSpec
    |
    v
CompilerConfig
    |
    v
CompiledProblemContract
    |
    +--> build_capability_store(feature_table)
    |
    +--> build_delay_store(feature_table, delay_seconds, runtime_metadata)
```

Capability stores are used for training. Delay stores are used for evaluating an artifact at a concrete delay. Runtime metadata carries calibrated assumptions from training to evaluation.

## What Compilers Must Decide

Compilers decide:

- how many past rows are context.
- which row is the decision anchor.
- where future candidate windows start and end.
- the maximum candidate slot width carried by the problem store.
- what feature prerequisites and warmup rows are needed.
- what runtime metadata must be persisted.

They should not decide action masks, prediction target batches, model architecture,
loss functions, training metrics, or evaluator aggregation.

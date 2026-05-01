# Temporal Architecture

## Purpose

`temporal` lowers feature tables into supervised temporal problem stores. It owns problem compilers, execution policies, input-normalization contracts, problem-store shapes, and temporal semantics.

Many time-series ML bugs are leakage bugs. Leakage happens when training examples include future information. Temporal contracts make time explicit.

## Core Row Concepts

```text
context_start        anchor / candidate_start        candidate window end
     |                         |                         |
     v                         v                         v
-----+-------------------------+-------------------------+---- time
     <---- model context -----> <---- candidate window --->
```

For current-row style problems, the anchor and candidate start may be the same row. The model may use context and anchor-available features. Prediction chooses an offset inside the candidate window. The execution policy maps that offset to an outcome row and compares it to baseline or optimum behavior.

## Problem Store

All compilers lower to `CompiledProblemStore`:

```text
feature_matrix
log_base_fees
timestamps
context_start_rows
anchor_rows
candidate_start_rows
candidate window end row array
max_candidate_slots
```

Prediction and evaluation consume this generic shape instead of compiler-specific details.
Problem stores own row geometry, not action availability. Action validity belongs to the execution policy because overflow and deadline behavior are policy semantics.

## Compiler Flow

```text
ProblemSpec + FeatureContract + ExecutionPolicyContract
        |
        v
problem compiler
        |
        +--> capability store for training
        |
        +--> delay store for evaluation at a concrete delay
        |
        v
CompiledProblemContract
```

Compilers publish feature prerequisites and runtime metadata codecs. Dataset builders and workflows call compiler contracts; they should not inspect concrete compiler classes.

## Execution Policy

Prediction chooses an action. Execution policy defines what that action means in the problem:

```text
execution policy + problem store + sample positions
        |
        +--> action mask
        |
decoded offsets
        |
        v
execute selected rows
        |
        +--> realized rows
        +--> baseline rows
        +--> optimum rows
        +--> overflow mask
```

Evaluators receive an execution-policy contract, not a policy id string.

## Extension Points

Add a compiler when example construction changes. Add a execution policy when outcome semantics change. Add input normalization when scaler fitting policy changes. Keep runtime metadata typed and routed through owner registries.

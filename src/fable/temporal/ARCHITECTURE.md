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
Problem stores own generic row geometry. Construction validates row-aligned feature, fee, and timestamp arrays; monotonic timestamps; aligned sample row arrays; positive action width; context rows inside each anchor; and non-empty candidate windows whose exclusive end stays inside the store. Store sample views also reject negative or out-of-range sample indices instead of relying on NumPy wrapping. Dataset-builder selection policy, fixed-context filtering, and inference timestamp-window sample filtering live in corpus preparation, not in the generic store.

Problem stores do not own action availability. `max_candidate_slots` is the action width, not a guarantee that every physical candidate window has that many rows. Action validity belongs to the execution policy because overflow and deadline behavior are policy semantics.

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

Training compiles the maximum supported delay into a capability store and a **Temporal Capability**. The capability is the artifact-facing runtime value that carries compiler runtime metadata, maximum delay, and action width into inference. Evaluation defaults and delay checks use the artifact Temporal Capability as authority; `ProblemSemantics.max_delay_seconds` remains authored problem provenance. Evaluation compiles a concrete delay store from the capability; it does not rediscover action width from the evaluation corpus. Storage artifact codecs own the persisted Temporal Capability envelope.

## Execution Policy

Prediction chooses an action. Execution policy defines what that action means in the problem:

```text
execution policy + problem store + sample indices
        |
        +--> prepared Action Space
             +--> sample indices
             +--> action width
             +--> action mask
        |
        +--> prepared temporal facts
             +--> Action Space
             +--> Temporal Outcome Facts
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

Model-input representation and prediction target construction consume the same prepared Action Space. Prediction training consumes prepared temporal facts so training state and target batches share one per-split policy preparation. Evaluators receive an execution-policy contract, not a policy id string.

## Extension Points

Add a compiler when example construction changes. Add a execution policy when outcome semantics change. Add input normalization when scaler fitting policy changes. Keep runtime metadata typed, routed through compiler registries, and bundled into Temporal Capability at artifact boundaries.

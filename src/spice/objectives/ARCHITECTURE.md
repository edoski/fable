# Objectives Architecture

## Purpose

`objectives` defines what optimization and tuning should maximize or minimize.

## Theory

An objective can be a validation metric from the training loop or a benchmark metric from evaluator scoring. Separating objective selection from prediction and evaluation lets you train one prediction task while tuning for a decision-quality metric.

## Invariants

Validation objectives read validation metrics directly. Evaluation objectives declare the benchmark metric to optimize. Objective configs must match the selected evaluation when they benchmark an evaluator. Model-bound metric production belongs to the **Objective Runtime** in `modeling.objective_runtime`.

Objective contracts consume the shared metric ABI from `spice.metrics`; they do not own metric result shapes.

## Extension Points

Add an objective type when optimization policy changes. Do not put evaluator execution in objectives or tuning code; route model-bound metric production through Modeling.

## Objective Flow

```text
training epoch
    |
    +--> validation metrics --------+
    |                               |
    +--> Objective Runtime ----------+
                                    |
                                    v
                              objective metric
                                    |
                                    v
                         early stopping / tuning direction
```

## Beginner Context

The objective is the scalar value used to compare model states or tuning trials. A metric can be useful for diagnostics without being the objective. For example, an evaluator may report many economic metrics, but tuning needs one primary metric and one direction.

## Evaluation Objective Boundary

Objectives are policy-only: metric id, direction, benchmark binding, and semantics. Evaluation-backed objective metrics use `modeling.objective_runtime`, which calls `modeling.scoring` for the generic sequence:

```text
check decoded-result id -> predict -> evaluate -> return metrics
```

This keeps Optuna and training loops from knowing evaluator internals.

Config-facing objective coercion normalizes invalid payload envelopes to `ConfigResolutionError` and returns already typed objective configs unchanged.

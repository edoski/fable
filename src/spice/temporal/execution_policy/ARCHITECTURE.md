# Execution Policy Architecture

## Purpose

`temporal.execution_policy` defines which actions are available and how a selected candidate offset becomes a realized outcome.

## Theory

Prediction chooses an action. Training and inference need to know which actions the policy can resolve, and evaluation needs to know what the selected action would have cost under the problem definition. An execution policy is this bridge from action space to outcome.

## Invariants

Policies must be compiled contracts, not workflow branches. Config-facing payload errors use `ConfigResolutionError`. Evaluators receive an execution-policy contract and a decoded prediction result, then compute metrics from the problem store.

The local policy table owns ids, concrete config types, and compile hooks. `core.specs` supplies only the mechanical owner-spec helper for payload coercion and compile-time type assertions.

The prepared Action Space is the policy-owned alignment object for selected samples. It carries sample indices, action width, and the action mask used by model-input representation, prediction targets, decoding, and evaluator replay. The compiled execution-policy contract validates selected-sample alignment, store action width, and action-mask shape when preparing it.

## Extension Points

Add a policy for a new action-to-outcome rule. Keep policy metadata explicit so artifacts can be understood later.

## Flow

```text
decoded action offset
        +
problem store candidate-window rows
        |
        v
realized fee / outcome
        |
        v
evaluator metric calculation
```

## Beginner Context

The model's output is not automatically an economic result. It is an action. The execution policy defines how that action interacts with the temporal problem. This keeps "what did the model choose?" separate from "what happened because of that choice?"

## Contract Rule

Model-input representation, prediction target construction, and evaluators receive an execution-policy contract, not a policy id string. Dataset preparation owns when selected samples become prepared Action Space or prepared temporal facts. Runtime and evaluation consume those prepared facts, so selected-sample alignment is validated at the execution-policy seam rather than re-derived in Batch Plan or evaluator scoring.

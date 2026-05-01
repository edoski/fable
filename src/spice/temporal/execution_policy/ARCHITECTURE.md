# Execution Policy Architecture

## Purpose

`temporal.execution_policy` defines which actions are available and how a selected candidate offset becomes a realized outcome.

## Theory

Prediction chooses an action. Training and inference need to know which actions the policy can resolve, and evaluation needs to know what the selected action would have cost under the problem definition. An execution policy is this bridge from action space to outcome.

## Invariants

Policies must be compiled contracts, not workflow branches. Config-facing payload errors use `ConfigResolutionError`. Evaluators receive an execution-policy contract and a decoded prediction result, then compute metrics from the problem store.

## Extension Points

Add a policy for a new action-to-outcome rule. Keep policy metadata explicit so artifacts can be understood later.

## Flow

```text
decoded action offset
        +
problem store candidate rows
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

Model-input representation, prediction target construction, and evaluators receive an execution-policy contract, not a policy id string. That contract exposes action masks, target behavior, realization behavior, and semantics without requiring callers to know concrete policy classes.

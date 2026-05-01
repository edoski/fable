# Config Architecture

## Purpose

`config` turns human-facing YAML and CLI selections into typed runtime configs. It is the edge where names become objects. Downstream code should receive explicit, resolved config models instead of resolving strings again.

## Mental Model

```text
surface YAML + workflow selection
          |
          v
surface frame with selection overrides
          |
          v
named YAML groups loaded from conf/
          |
          v
owner coercers rebuild concrete nested configs
          |
          v
AcquireConfig / TrainConfig / TuneConfig / EvaluateConfig
```

Configuration has two jobs:

```text
composition  choose named pieces and apply selection overrides
validation   turn those pieces into typed models with clear errors
```

The config package should not train models, evaluate predictions, acquire blocks, or write artifacts. It resolves intent.

## Workflow Selection

A workflow selection is unresolved workflow intent. CLI commands and benchmark plans build selections from user-facing choices. Resolution applies the selection to a surface, then loads named config groups and owner coercers to produce a workflow config.

Selections usually refer to problem specs by name. Benchmark problem grids may supply an inline `ProblemSpec`; this still uses the same resolution path, and the resolved workflow config stores the full executable problem.

## Owner Coercion

Generic Pydantic validation cannot always reconstruct concrete nested types. The owning package knows which selector maps to which concrete type:

```text
evaluation.engine        -> evaluation config type
dataset_builder.id       -> dataset-builder config type
model.id                 -> model-family config type
problem.compiler.id      -> problem compiler config type
problem.execution_policy.id   -> execution-policy config type
```

So config resolution and snapshot loading call owner coercers. This keeps all implementation-selection knowledge in the owning domain.

## Resolved Workflow Snapshot Codec

Remote execution, benchmark run persistence, and tuned-parameter reapplication do not re-resolve surfaces. They serialize or load an already resolved config snapshot:

```text
resolved Train/Tune/Evaluate config
        |
        v
workflow_config_snapshot_json()
        |
        v
remote runner, benchmark plan JSONL, or tuned-param materialization
        |
        v
hydrate_workflow_config_snapshot()
        |
        v
owner coercers reconstruct concrete nested configs
```

The snapshot codec is intentionally resolved-snapshot-only. It supports train, tune, and evaluate. Acquire has different acquisition/provider concerns and is resolved through normal workflow resolution.

## Public API Boundary

`spice.config` exports resolved config types, workflow selection types, `resolve_workflow_config()`, workflow snapshot codecs, and config-owned coercers such as `coerce_problem_spec()` and `coerce_features_config()`.

It does not re-export dataset-builder coercion. Dataset-builder config coercion belongs to `spice.modeling.dataset_builders`.

## Default Remote Target

The CLI owns the convenience default:

```text
DEFAULT_REMOTE_TARGET = "disi_l40"
```

Resolved configs and execution APIs do not carry a hidden fallback target. Submitted commands pass an explicit target name downstream.

## Extension Points

Add a public config group when users need to name, inspect, or edit that spec. Add a workflow selection field only when it is legitimate unresolved workflow intent or runtime control. Add concrete implementation selection in the owner package, not in config.

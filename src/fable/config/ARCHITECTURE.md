# Config Architecture

## Purpose

`config` turns human-facing YAML and CLI selections into typed runtime configs. It is the edge where names become objects. Downstream code should receive explicit, resolved config models instead of resolving strings again.

## Mental Model

```text
surface YAML + produce-root workflow selection
          |
          v
workflow-shaped surface refs
          |
          v
named YAML groups loaded from conf/
          |
          v
owner coercers rebuild concrete nested configs
          |
          v
AcquireConfig / TrainConfig / TuneConfig

root-id evaluate selection
          |
          v
named evaluator/runtime controls
          |
          v
EvaluateConfig
```

Configuration has two jobs:

```text
composition  choose named pieces and apply selection overrides
validation   turn those pieces into typed models with clear errors
```

Concrete config models inherit the shared strict `core.config_model.ConfigModel` base. The config package and owner packages own their fields and coercers; `core` only owns the generic Pydantic policy.

The config package should not train models, evaluate predictions, acquire blocks, or write artifacts. It resolves intent.

## Workflow Selection

A workflow selection is unresolved workflow intent. CLI commands build selections from user-facing choices, and benchmark materialization may build them internally while producing durable plan entries. Produce-root workflows apply the selection to a surface, then load named config groups and owner coercers to produce a workflow config. Evaluate fresh resolution uses exact root ids plus evaluator/runtime controls instead of composing a surface.

Selections usually refer to problem specs by name. Benchmark problem grids may supply an inline `ProblemSpec`; this still uses the same resolution path, and the resolved workflow config stores the full executable problem.

`config.selections` owns workflow selection types and generic workflow field metadata. CLI parsing, fresh resolution, and benchmark materialization consume that metadata instead of each restating allowed workflow-selection fields. Benchmark-specific dimension names stay with benchmark schema; root-vs-coordinate selection policy stays with benchmark plan materialization.

## Config Group Loading

Config groups have two loading interfaces. `groups.load_named_group_payload()` returns a canonical raw dict for CLI show/edit, templates, fixture mutation, benchmark raw specs, and durable YAML output. `typed_groups.load()` reads group documents directly and returns context-free typed named groups for workflow resolution and other internal runtime setup.

`group_catalog` owns group names, directories, identity checks, and validators shared by the raw and typed interfaces. `typed_groups` exposes typed views of that catalog metadata; casts stay local there. Owner packages own concrete dispatch inside a group. Tuning-space loading stays in resolution because it depends on the selected model and problem. Benchmark typed loading stays in benchmarks.

Fresh resolution loads the named surface directly from the concrete Workflow Selection and constructs acquire/train/tune configs from the selected Surface values plus overrides. Selection overrides coalesce only on `None`; empty refs are treated as explicit invalid refs instead of falling back to surface defaults. `surfaces` owns frame models only. Fresh resolution and snapshot hydration share final field policy and config assembly through `resolved_workflows.py`, but fresh resolution does not route through snapshot hydration.

## Owner Coercion

Generic Pydantic validation cannot always reconstruct concrete nested types. The owning package knows which selector maps to which concrete type:

```text
evaluator.id             -> evaluator config type
model.id                 -> model-family config type
problem.compiler.id      -> problem compiler config type
problem.execution_policy.id   -> execution-policy config type
```

So config group typed loading, resolved snapshot hydration, and typed tuned-parameter transforms call owner coercers. Config-facing coercer envelope errors normalize to `ConfigResolutionError`. Owner tables stay in their packages; `core.specs` only supplies mechanical helpers for payload/id extraction, concrete-type validation, and compile-time type assertions.

A Concrete Owner Config is the concrete local-spec config selected by an owner id. Owner coercers preserve identity only for Concrete Owner Config instances. Abstract typed selector configs are redispatched by id and validated as the concrete type before runtime code sees them. This keeps all implementation-selection knowledge in the owning domain.

## Surface Resolution

Surface resolution is the fresh path from produce-root Workflow Selection to Workflow Config. It applies the selection to the Surface, loads named config groups from the resulting workflow-shaped refs, calls owner coercers, and instantiates `AcquireConfig`, `TrainConfig`, or `TuneConfig` from already typed pieces. Evaluate fresh resolution instantiates `EvaluateConfig` from a Root Consumer Selection plus evaluator/runtime controls.

Surface resolution does not hydrate raw resolved snapshots. Resolved snapshots are already past selection and surface ownership.

Required workflow refs are checked during fresh resolution: acquire requires
acquisition/provider refs, model workflows require model/training/split refs, and
tune and tuned-train require tuning-space refs. Evaluate requires an artifact id,
corpus id, evaluator, and evaluation window.

## Resolved Workflow Hydration

Remote execution and benchmark run persistence do not re-resolve surfaces. They serialize or load an already resolved config snapshot:

```text
resolved Train/Tune/Evaluate config
        |
        v
workflow_config_snapshot_json()
        |
        v
remote runner or benchmark plan JSONL
        |
        v
hydrate_workflow_config_snapshot()
        |
        v
owner coercers reconstruct concrete nested configs
```

Resolved Workflow Hydration is intentionally snapshot-only. It supports train, tune, and evaluate. Acquire has different acquisition/provider concerns and is resolved through Surface resolution.

Tuned-parameter application is a typed transform. It rebuilds `TrainConfig` or `TuneConfig` from validated concrete nested configs and does not call snapshot hydration.

## Public API Boundary

`spice.config` exports resolved config types, workflow selection types, `resolve_workflow_config()`, workflow snapshot codecs, and config-owned coercers such as `coerce_problem_spec()` and `coerce_features_config()`.

The generic `ConfigModel` base is intentionally not a public `spice.config` export. Modules that define config models import it from `spice.core.config_model`.

## Default Remote Target

The CLI owns the convenience default:

```text
DEFAULT_REMOTE_TARGET = "disi_l40"
```

Resolved configs and execution APIs do not carry a hidden fallback target. Submitted commands pass an explicit target name downstream.

## Extension Points

Add a public config group when users need to name, inspect, or edit that spec. Add a workflow selection field only when it is legitimate unresolved workflow intent or runtime control. Add concrete implementation selection in the owner package, not in config.

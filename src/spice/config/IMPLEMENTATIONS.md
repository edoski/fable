# Concrete Config Resolution

Config resolution turns small named YAML fragments into fully typed workflow configs. The concrete implementation is strict: every local spec is validated against the model for its own group entry before workflow code sees it.

## Mental Model

The repository separates three concepts:

```text
authored YAML
  -> resolved mapping
  -> typed Pydantic model
  -> compiled runtime contract
```

YAML is user-facing. Pydantic models are the internal boundary. Concrete config models inherit the strict `core.config_model.ConfigModel` base; implementation packages own their fields and owner coercers. Compiled contracts are executable objects used by modeling, acquisition, and evaluation.

A Concrete Owner Config is the concrete local-spec config selected by an owner id. Owner coercers may preserve that object by identity. Abstract typed selector configs are redispatched by id and validated against the concrete type before runtime modules receive them.

## Local Specs

Local specs are small named configs such as `model/lstm.yaml`, `problem/current_row_nominal.yaml`, `evaluation/poisson_replay_2h.yaml`, or `evaluation/full_temporal_replay.yaml`.

Config Group Loading has two explicit paths:

```text
groups.load_named_group_payload()
  -> canonical raw dict for show/edit/template/benchmark raw specs

typed_registry.load_model_config(), typed_registry.load_problem_spec(), ...
  -> typed owner config for resolution/runtime callers
```

Resolution uses the same typed pattern across registries:

```text
mapping contains id or name
  -> lookup local spec
  -> require matching config type
  -> compile contract
```

This gives concrete implementations one owner. For example, the evaluation package owns evaluator config models, and the modeling package owns dataset-builder config models.

## Surface Resolution

A surface is a named bundle of config choices. It resolves into a complete workflow frame:

```text
surface: current_row_fee_dynamics
overrides: model=lstm_icdcs_2026, delay_seconds=36

        surface YAML
             |
             v
  chain/dataset/provider/problem
  features/model/prediction
  objective/evaluation/training/split
             |
             v
      workflow config model
```

`selection_application` replaces selected acquire/train/tune surface fields before final resolution. Evaluate remains root-id based and does not use surface composition. The result is a typed acquire, train, tune, or evaluate config.

Surface resolution is a typed construction path. Once named groups and overrides have been resolved, `resolution.py` instantiates the workflow config from typed pieces. It does not round-trip through raw resolved snapshot hydration.

Command workflow resolution starts from sparse operator values. `workflow_selection_from_values()` drops unsupported and unset fields for the selected workflow, validates the matching `WorkflowSelection`, and `resolve_workflow_command_config()` resolves that selection into the concrete workflow config. CLI commands pass option dictionaries to this config-owned path instead of constructing selection models locally.

## Resolved Workflow Hydration

Resolved snapshot hydration is the raw-payload path for already materialized train, tune, and evaluate configs. It validates the snapshot workflow marker, then uses owner coercers to rebuild concrete nested configs.

Acquire is excluded from snapshot hydration. Acquire configs contain provider/acquisition concerns and are produced through surface resolution.

Tuned-parameter application is not snapshot hydration. It is a typed train/tune config transform that applies tuned params through owner-family validators and rebuilds the workflow config directly.

Model workflow snapshots contain training/evaluation fields: chain, dataset, problem, model, dataset builder, features, prediction, objective, optional evaluation, storage, artifact, split, training, study, tuning, and tuning space depending on workflow.

Evaluation date expands into concrete UTC windows:

```text
dataset.evaluation_date
  -> evaluation_start: midnight UTC
  -> evaluation_end: next midnight UTC
  -> history_end: evaluation_start
```

## Objective And Evaluation Rules

Training and tuning can optimize either validation metrics or evaluator metrics.

```text
validation objective
  -> use shared MetricSet directly

evaluation objective
  -> compile named evaluator
  -> score validation samples through evaluator
  -> optimize evaluator metric
```

For train and tune, an evaluation objective must name the same benchmark as the selected evaluation config. Evaluate workflow can run a diagnostic evaluator directly; the artifact still validates against the training semantics stored in its manifest.

Runtime evaluator contracts carry typed evaluator configs. Persisted evaluation summaries carry immutable Evaluation Config Snapshots, and artifact storage codecs serialize and hydrate those snapshots at the persisted-state boundary.

## Config Boundary Errors

Config-facing errors are reported as `ConfigResolutionError`. This keeps YAML mistakes distinct from runtime data errors.

Typical failures:

| Failure | Meaning |
| --- | --- |
| Unknown spec id | A YAML reference names no checked-in spec. |
| Wrong local config type | A spec was routed to a compiler that does not own it. |
| Extra fields | The YAML contains fields not accepted by that concrete config. |
| Missing owner id | Registry dispatch cannot choose an implementation. |
| Objective benchmark mismatch | Training would optimize a different evaluator than requested. |

## CLI Remote Target Boundary

The CLI command layer owns the default remote target:

```text
spice train
  -> target defaults to disi_l40
  -> execution config resolves explicitly
  -> downstream submit receives target name
```

Execution and transfer code do not guess a remote target. That keeps command ergonomics centralized and lower layers deterministic.

## Extension Pattern

To add a concrete config type, subclass the shared config base in the package that owns the implementation, register the local spec id, and make the compiler require that concrete model. The config package should resolve names and load resolved workflow snapshots; implementation packages should own implementation-specific fields.

# Concrete Config Resolution

Config resolution turns small named YAML fragments into fully typed workflow configs. The concrete implementation is strict: every local spec is validated against the model for its own registry entry before workflow code sees it.

## Mental Model

The repository separates three concepts:

```text
authored YAML
  -> resolved mapping
  -> typed Pydantic model
  -> compiled runtime contract
```

YAML is user-facing. Pydantic models are the internal boundary. Compiled contracts are executable objects used by modeling, acquisition, and evaluation.

## Local Specs

Local specs are small named configs such as `model/lstm.yaml`, `problem/current_row_nominal.yaml`, `evaluation/poisson_replay_2h.yaml`, or `evaluation/full_temporal_replay.yaml`.

Resolution uses the same pattern across registries:

```text
mapping contains id/engine/family
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

Overrides replace selected surface fields before final resolution. The result is a typed acquire, train, tune, or evaluate config.

## Workflow Hydration

Acquire configs contain corpus acquisition fields: chain, dataset, provider, acquisition, features, and problem.

Model workflow configs contain training/evaluation fields: chain, dataset, problem, model, dataset builder, features, prediction, objective, optional evaluation, storage, artifact, split, training, study, tuning, and tuning space depending on workflow.

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
  -> use validation MetricSet directly

evaluation objective
  -> compile named evaluator
  -> score validation samples through evaluator
  -> optimize evaluator metric
```

For train and tune, an evaluation objective must name the same benchmark as the selected evaluation config. Evaluate workflow can run a diagnostic evaluator directly; the artifact still validates against the training semantics stored in its manifest.

## Config Boundary Errors

Config-facing errors are reported as `ConfigResolutionError`. This keeps YAML mistakes distinct from runtime data errors.

Typical failures:

| Failure | Meaning |
| --- | --- |
| Unknown spec id | A YAML reference names no checked-in spec. |
| Wrong local config type | A spec was routed to a compiler that does not own it. |
| Extra fields | The YAML contains fields not accepted by that concrete config. |
| Missing engine/family/id | Registry dispatch cannot choose an implementation. |
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

To add a concrete config type, add the Pydantic model in the package that owns the implementation, register the spec id or engine, and make the compiler require that concrete model. The config package should resolve names and load resolved workflow snapshots; implementation packages should own implementation-specific fields.

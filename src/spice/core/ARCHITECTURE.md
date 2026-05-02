# Core Architecture

## Purpose

`core` contains primitives that are smaller than any domain package: error types, filesystem atomicity, local spec lookup helpers, the shared strict config model base, validation helpers, reporting, constants, and metric rendering.

It should stay boring. If a helper knows what an evaluator, model, corpus, or workflow means, it probably belongs outside `core`.

## Spec Helpers

`core.config_model.ConfigModel` owns the shared Pydantic config defaults used by config-facing models: extra fields are rejected and assignment is validated. Domain packages own their concrete fields and owner coercers; `core` owns only the boring base policy.

Local implementation registries use three core helpers:

```text
require_mapping_id(payload, field_label)
  -> validate that a mapping has a non-empty "id"

owner_payload(payload, owner, config_type)
  -> normalize a config-facing mapping/config envelope or raise ConfigResolutionError

owner_payload_id(payload, owner, config_type, id_label)
  -> normalize the envelope and require its id in one step

lookup_local_spec(specs, spec_id, field_label)
  -> select a local spec or raise a clear ConfigResolutionError

require_spec_config(config, config_type, label)
  -> assert dispatch received the concrete config type selected by the spec
```

The pattern is:

```text
payload -> owner coercer -> concrete config
config.id -> local spec
spec.config_type + config -> require_spec_config
concrete config -> compile hook
```

This keeps validation at the config boundary and keeps compile dispatch simple. It also avoids defensive serialize/revalidate wrappers that hide architecture drift.

Owner coercers return already typed config objects unchanged. Raw payload handling stays at the owner coercer edge, and invalid config-facing envelopes become `ConfigResolutionError`.

## Error Vocabulary

```text
ConfigResolutionError  user/config selection or YAML or snapshot resolution problem
StateLayoutError       malformed persisted state or root-kind mismatch
MissingStateError      expected persisted state is absent
SpiceOperatorError     operational failure reported to the CLI/user
StateConflictError     safe write would overwrite disallowed existing state
```

Using the right error type matters because command layers format these failures differently and tests assert specific failure policies.

## Filesystem Primitives

`core.files` owns atomic file and path replacement mechanics. Storage builds higher-level staging and partial-commit primitives on top of these functions.

```text
write_path_atomic(path, writer)
replace_path_atomic(target, source, replace=...)
replace_paths_atomic([(target, source), ...], replace=...)
remove_path(path)
prune_empty_directories(path, stop_at=...)
```

Callers choose the explicit replacement policy at the call site or through storage staging primitives. The low-level helpers stay policy-neutral.

## Dependency Rule

```text
core -> standard library / third-party basics only
domain packages -> core
workflows -> domain packages
cli -> workflows and services
```

`core` must not import modeling, evaluation, storage roots, workflows, CLI, or concrete implementation packages.

# Core Architecture

## Purpose

`core` contains primitives that are smaller than any domain package: error types, filesystem atomicity, local spec lookup helpers, validation helpers, reporting, constants, and metric rendering.

It should stay boring. If a helper knows what an evaluator, model, corpus, or workflow means, it probably belongs outside `core`.

## Spec Helpers

Local implementation registries use three core helpers:

```text
require_mapping_id(payload, field_label)
  -> validate that a mapping has a non-empty "id"

lookup_local_spec(specs, spec_id, field_label)
  -> select a local spec or raise a clear ConfigResolutionError

require_spec_config(config, config_type, label)
  -> assert dispatch received the concrete config type selected by the spec
```

The pattern is:

```text
payload -> owner coercer -> concrete config
config.id / config.engine -> local spec
spec.config_type + config -> require_spec_config
concrete config -> compile hook
```

This keeps validation at the config boundary and keeps compile dispatch simple. It also avoids defensive serialize/revalidate wrappers that hide architecture drift.

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

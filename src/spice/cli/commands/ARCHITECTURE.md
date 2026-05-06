# CLI Commands Architecture

## Purpose

`cli.commands` contains command modules grouped by operator intent: workflow execution, config inspection, Benchmark Run operations, storage inspection, and transfer.

## Theory

Command modules should translate user language into typed selections or storage selectors for the owning package boundary. They should not reimplement package internals. This keeps command behavior predictable and makes the Python APIs useful outside the terminal.

## Invariants

Workflow commands construct typed Workflow Selection models and call `resolve_workflow_config()` once. Storage commands map options into storage selectors and render Storage Operator Outcomes. Transfer commands use storage selectors over existing catalog records. Config commands use config group APIs. Command modules may format output, but persistence, match policy, and ML behavior remain elsewhere.

Remote workflow commands and transfer commands share the same remote-target option. The default target is provided by the CLI layer, then passed downstream explicitly.

## Extension Points

Create a new command module when the user intent is distinct. Prefer small command functions that delegate to one package API over command functions that coordinate many low-level modules.

## Command Map

```text
commands/
  workflows.py  local acquire and remote train/tune/evaluate
  config.py     config list/show/edit commands
  benchmark.py  Benchmark Run planning/submission/collection/indexing
  storage.py    show/delete/refresh storage commands
  transfer.py   push/pull storage roots
```

## Selector Rule

Workflow commands use **Workflow Selections**. Storage and transfer commands use catalog selectors:

```text
Workflow Selection -> build future work
catalog selector   -> find existing persisted root
```

Do not mix these concepts. A storage command should not need a surface. A workflow command should not need a catalog record unless it is loading existing artifact or study state through workflow config. Storage command modules do not decide show/delete ambiguity; they print storage-owned outcomes and translate narrowing attributes to CLI flag names.

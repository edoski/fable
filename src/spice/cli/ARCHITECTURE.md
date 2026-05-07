# CLI Architecture

## Purpose

`cli` is the operator edge. It translates command-line options into workflow selections, config-inspection actions, storage inspection/deletion, Benchmark Run operations, and remote transfer operations.

The CLI may provide user conveniences. It should not own model training, evaluator logic, storage layout internals, or resolved workflow snapshot rules.

## Command Flow

```text
operator command
    |
    v
Typer option parsing
    |
    v
workflow selection / storage selector
    |
    v
config resolution or storage service
    |
    v
workflow/service result rendered to stdout
```

Workflow commands run acquisition locally and submit CUDA workflows to remote execution. Transfer commands call execution transfer services. Config commands call config group APIs.

`cli.workflow_command_selection` constructs `AcquireWorkflowSelection`, `TrainWorkflowSelection`, `TuneWorkflowSelection`, or `EvaluateWorkflowSelection` from explicit CLI values before commands run local acquire or hand a resolved workflow config to execution submission. Config owns fresh resolution from typed Workflow Selection to Workflow Config; execution serializes resolved snapshots for remote runs.

CLI command registration adapts `SpiceOperatorError` into Typer/Click operator errors at the command seam. Core errors stay plain project exceptions; parse errors remain Typer-owned.

## Remote Target Boundary

```text
CLI option default:
  DEFAULT_REMOTE_TARGET = "disi_l40"

Downstream APIs:
  open_execution_session(target_name)
  transfer(..., session=...)
```

The default exists once, at the CLI edge. This lets the common cluster stay convenient without making lower layers cluster-aware.

## Shared Options

`cli.options` owns common option aliases such as filters, storage-root options, and `RemoteTargetOption`. Workflow submit commands and transfer commands use the same target option so help text and behavior stay consistent.

Within workflow commands, generic execution-panel options share one helper. Selection options, output options, and execution options are distinct because they represent different operator decisions.

## Local Versus Remote Runs

```text
local acquire:
  may use --storage-root
  resolves config and runs acquisition in current process

remote train/tune/evaluate:
  may use --target, --dependency, --detach
  do not expose --submit or --storage-root
  remote storage comes from the execution target spec
  execution submission follows logs when the target allows it unless --detach is set
```

This matches the CUDA operating model: corpora are acquired locally and pushed, while train, tune, and evaluate execute on the cluster.

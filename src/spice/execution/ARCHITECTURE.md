# Execution Architecture

## Purpose

`execution` owns remote target models, **Execution Session** creation, direct workflow submit/follow lifecycle, SSH command execution, Slurm submission, remote workflow entrypoints, and SSH/rsync transfer orchestration. It receives explicit target names from operator edges.

## Flow

```text
CLI submit command
    |
    v
resolve workflow config locally
    |
    v
serialize resolved config JSON
    |
    v
ssh + sbatch remote command
    |
    v
python -m spice.execution.remote_runner <task> <config_json>
    |
    v
hydrate_workflow_config_snapshot_json()
    |
    v
train/tune/evaluate workflow
```

Remote execution does not re-resolve surfaces. The local process chooses all named config pieces first. The remote process receives the fully resolved snapshot and reconstructs concrete nested config types through owner coercers.

## Target Rule

Execution sessions require an explicit target name. The CLI supplies `DEFAULT_REMOTE_TARGET` when an operator does not pass `--target`.

```text
CLI default allowed here:
  spice train

No hidden default here:
  session = open_execution_session(target_name)
```

This keeps execution reusable for tests, scripts, and future target selection without a hard-coded cluster dependency in the service layer.

## Supported Resolved Workflows

The remote runner supports already resolved workflow snapshots:

```text
train
tune
evaluate
```

Acquire is not routed through this resolved workflow snapshot path. Acquisition has provider/RPC behavior and storage commit mechanics that are resolved through the normal acquire workflow path.

## Direct Workflow Submission

`execution.submission` owns direct train/tune/evaluate submission lifecycle: open the explicit target session, submit the resolved workflow, emit submission events, follow logs when target policy allows it, detach on operator interrupt, and raise on non-completed final states. Benchmark Plan Execution does not use this lifecycle because it submits persisted plans without following each job.

## Transfer Boundary

`execution.transfer_transaction` owns Storage Transfer Transaction behavior. A `StorageTransferTransaction` receives an **Execution Session**, resolves remote records through the **Remote Catalog Record Codec**, derives canonical source and destination locations through catalog materialization, stages roots through local or remote transfer adapters, invokes rsync, promotes with root-kind validation, and cleans failed stages without hiding the primary failure. Public callers use `push_dataset()` and `pull_artifact()` on the transaction.

The remote helper remains `spice.storage.sync_cli` because it performs local storage operations on the remote machine, not workflow execution. Its commands are machine-facing; public operators use `spice transfer push dataset` and `spice transfer pull artifact`.

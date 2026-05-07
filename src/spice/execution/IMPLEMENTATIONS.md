# Concrete Remote Execution

Remote execution submits train, tune, and evaluate workflows to a SLURM cluster over SSH. It also owns rsync-based transfer orchestration for storage roots. The checked-in target is `disi_l40`.

## Mental Model

The local CLI resolves config, then remote execution rewrites storage paths for the cluster and submits a batch job.

```text
local workflow config
  -> Execution Session for explicit target
  -> remote storage root rewrite
  -> sbatch script
  -> python -m spice.execution.remote_runner
```

Acquire runs locally. The concrete **Execution Session** supports train, tune, and evaluate submission.

## Target Config

An execution target defines:

| Field | Meaning |
| --- | --- |
| `ssh.host`, `ssh.user` | SSH destination. |
| `repo_root` | Repo path on the cluster. |
| `venv_root` | Python environment path on the cluster. |
| `storage_root` | Cluster output storage root. |
| `log_root` | Cluster log directory. |
| Workflow resources | SLURM resources per train/tune/evaluate. |
| `follow_by_default` | Whether CLI tails logs after submit. |

`disi_l40` is the current target spec. CLI commands default to this target only at the CLI layer.

## Execution Session

`spice.execution.session` is the target-bound Interface for remote operations. It owns shell quoting, command execution, module execution, rsync, SLURM submission, target follow policy, job following, final-state reads, and remote git commit lookup. `spice.execution.submission` owns the higher-level direct workflow submit/follow lifecycle. A job submission is only the submitted job receipt: workflow task, job id, and log path.

There is no additional remote-execution session protocol while SSH/SLURM is the only Adapter.

## SLURM Submission

The session builds an SSH command that runs `bash -lc` remotely. It renders an `sbatch` script, sources the remote environment, sets `PYTHONUNBUFFERED=1`, and runs:

```text
python -m spice.execution.remote_runner {task} {config_json}
```

`sbatch` output is parsed for the job id.

## Remote Runner

The remote runner is the cluster-side entrypoint. It receives the task name and serialized config JSON, loads the resolved workflow snapshot, and calls the workflow function.

```text
remote_runner train  '{...json...}'
remote_runner tune   '{...json...}'
remote_runner evaluate '{...json...}'
```

## Log Following

After direct workflow submit, `execution.submission` follows the remote log when `ExecutionSession.follow_by_default` allows it and `--detach` is absent. The session polls `squeue` while tailing output. When the job leaves the queue, it asks `sacct` for final state. Any final state other than `COMPLETED` is treated as an operator-facing error.

## Storage Root Rewrite

Before submission, the workflow config storage root is rewritten to the target's cluster storage root. This makes the same logical config run against local storage locally and cluster storage remotely.

## Transfer

`StorageTransferTransaction` pushes local dataset roots to a target and pulls remote artifact roots back locally through an **Execution Session**. The transaction owns prepare, rsync, promote, and cleanup for both directions; local and remote transfer adapters provide the storage-lifecycle operations on the side that receives the root. Cleanup is best-effort and preserves the primary rsync or promote failure as the raised error.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| SSH command fails | Cluster is unreachable or command errored. |
| `sbatch` output missing job id | Submission did not return expected SLURM text. |
| Final state not `COMPLETED` | Job failed, timed out, or was cancelled. |
| Unsupported task | The remote runner only handles train, tune, evaluate. |
| Missing target config | CLI target name does not resolve. |

## Extension Pattern

A second remote execution Adapter should keep the remote runner contract: typed workflow config in, workflow function out. Do not introduce a backend protocol until that second Adapter exists.

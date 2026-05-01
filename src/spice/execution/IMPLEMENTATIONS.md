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

`spice.execution.session` is the target-bound Interface for remote operations. It owns shell quoting, command execution, module execution, rsync, SLURM submission, job following, final-state reads, and remote git commit lookup.

There is no execution backend protocol while SSH/SLURM is the only Adapter.

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

After submit, the CLI can follow the remote log. The session polls `squeue` while tailing output. When the job leaves the queue, it asks `sacct` for final state. Any final state other than `COMPLETED` is treated as an operator-facing error.

## Storage Root Rewrite

Before submission, the workflow config storage root is rewritten to the target's cluster storage root. This makes the same logical config run against local storage locally and cluster storage remotely.

## Transfer

Transfer services push local dataset roots to a target and pull remote artifact roots back locally through an **Execution Session**. They resolve catalog records through the remote catalog codec, prepare a hidden staging root, run rsync, then finalize through storage lifecycle validation and catalog reindexing.

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

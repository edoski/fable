# ADR 0007: Native External Execution Boundary

## Status

Accepted.

## Context

FABLE (Fee Analysis through Blockchain Learning and Estimation) needs one narrow path from a workstation to a CUDA Slurm host. Native OpenSSH, Slurm, and file-transfer tools provide the host boundary.

## Decision

Remote submission uses cwd-local `REMOTE.yaml`, OpenSSH, a generated Slurm script, one `sbatch --parsable` call, and the returned positive numeric job ID. The script invokes the installed `fable` executable with a generated-job entry point. Workflow jobs receive one strict `WorkflowRequest` directly; candidate jobs receive one strict record containing the `TuneRequest` and Method.

Submission ends when Slurm returns the job ID. Scheduler tools monitor jobs, and file-transfer tools move completed objects between hosts.

## Consequences

The submission interface stays small. Scientific requests and durable objects remain independent of host, queue, log, and transfer state. `REMOTE.yaml` owns only connection and Slurm resource facts. The installed executable owns one fixed loader and Torch runtime profile and must remain unchanged while submitted jobs are queued.

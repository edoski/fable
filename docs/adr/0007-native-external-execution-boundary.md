# ADR 0007: Native External Execution Boundary

## Status

Accepted.

## Context

KAIROS needs one narrow path from a workstation to a CUDA Slurm host. Native OpenSSH, Slurm, and file-transfer tools provide the host boundary.

## Decision

Remote submission uses cwd-local `REMOTE.yaml`, OpenSSH, a generated Slurm script, one
`sbatch --parsable` call per allocation, and the returned positive numeric job ID. An allocation
contains either one process or an ordered batch of up to four independent processes. Every process runs
the same immutable Apptainer image through one exclusive Slurm step and receives exactly one GPU.
Its runscript invokes the installed `kairos` executable with a generated-job entry point. Workflow
processes receive one strict `WorkflowRequest` directly; candidate processes receive one strict
record containing the `TuneRequest` and Method index.

Submission ends when Slurm returns the job ID. Scheduler tools monitor jobs, and file-transfer tools move completed objects between hosts.

## Consequences

The submission interface stays small. Packing changes allocation efficiency, not scientific
execution: each fit or evaluation remains an isolated single-GPU process with its original
request, scratch, result, and resume behavior. Scientific requests and durable objects remain
independent of host, queue, log, and transfer state. `REMOTE.yaml` owns only connection, image,
storage, and Slurm resource facts. The immutable image owns one KAIROS revision plus its fixed
loader and Torch runtime profile.

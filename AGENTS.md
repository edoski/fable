## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for this repository; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use only `ready-for-agent`, `ready-for-human`, and `wontfix` as ordinary triage labels.

### Domain docs

Use terminology from `docs/CONTEXT.md`. If work conflicts with an existing ADR, state the conflict explicitly rather than silently overriding it.

### Verification tools

Use `uv run vulture` for dead-code checks. The repo config runs Vulture at 90% confidence; do not automatically assume reported code is dead. Manually verify every finding against dynamic usage, framework callbacks, validators, CLI registration, reflection, and config-driven references before deleting anything.

### Research image builds

The research login node does not expose Apptainer. Build immutable images through Slurm's
`sbuild` partition from an isolated checkout at
`/scratch.hpc/edoardo.galli3/build/fable-cuda-<short-sha>`. Assert the checkout's full commit SHA,
use `/scratch.hpc/edoardo.galli3/cache/fable-cuda-<short-sha>` as `APPTAINER_CACHEDIR`, and write
`/scratch.hpc/edoardo.galli3/deployments/fable-cuda-<short-sha>.sif`. The proven allocation is one
node and task, 8 CPUs, 30 GB RAM, and a one-hour limit. Run `apptainer build` followed by
`apptainer test`; preserve the preceding image until the replacement passes GPU smoke tests.

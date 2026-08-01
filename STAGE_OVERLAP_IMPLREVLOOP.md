# Stage overlap implementation ledger

## Authority and run policy

- User approval: start context-sensitivity work as soon as its nine full-feature parents exist, while the remaining feature-ablation cells finish.
- Design standard: clean break; smallest authored-roster interface; no compatibility layer, campaign-specific IDs, row skips, private-result reads, generic lifecycle framework, or unrelated cleanup.
- Checkout policy: work directly on `main`; no branch or worktree creation; one writer at a time; no push or pull request.
- Workflow: one implementation slice, one independent two-axis review, correction loop only if findings exist.
- This ledger is temporary and will be deleted after the slice is green and integration checks pass.

## Pre-run state

- Branch: `main`
- Immutable baseline: `5364af1dbfb01aa52397adad9d9bca7f29855409`
- Worktrees: only `/Users/edo/dev/python/fable` on `main`
- Pre-existing branch, not run-owned: `codex/compact-cuda-execution`
- Pre-existing untracked paths, not run-owned: `docs/experiments/feature_ablation.md`, `docs/research/inference-benchmark-implrevloop.md`, `docs/research/macos-inference-energy.md`
- Concurrent untracked path observed after baseline, not run-owned: `docs/research/inference-benchmark-slice-research.md`
- Governing sources: `AGENTS.md`, `docs/CONTEXT.md`, `docs/FABLE.md`, `docs/adr/0006-direct-durable-object-authority.md`, `docs/adr/0007-native-external-execution-boundary.md`

## Approved decisions

- A finalized experiment manifest remains the sole authority after closure.
- Before closure, the active hidden bundle's authored `cells.tsv` may supply the exact cell-to-record roster. Consumers still load only canonical durable records.
- Context-sensitivity preparation needs exactly the nine canonical `*.full` feature-ablation Studies. It does not need the other 93 ablation Studies or a finalized 102-cell feature manifest.
- Missing required full Studies fail through canonical `load_study`; do not add duplicate readiness state or defensive orchestration machinery.
- Permit up to four independent one-GPU processes in one allocation. Preserve process isolation, ordered inputs, one GPU per process, restart-safe `jobs.tsv`, and direct Slurm submission.
- Use a general balanced allocation split that minimizes allocation count and avoids singleton tails when possible.
- Research resources become 32 CPUs and 64 GiB per fit. Four fits request 128 CPUs and 256 GiB, within every approved four-GPU node and far above observed 2.1–3.1 GiB peak RSS.
- Approved GPU partitions: `h100sxm5,h100pcie,a100,l40s,l40`.
- Keep the current production image for live jobs. No image build is part of this run.
- Do not relax later scientific gates here. In particular, held-out evaluation stays sealed until all selected upstream decisions are frozen.

## Slice 1 — authored parents and four-fit allocations

### Expected outcome

Context-sensitivity preparation can begin from an open feature-ablation experiment once its nine authored full-feature Studies are canonical. Candidate and workflow launchers can pack up to four independent fits using the approved research resources, without changing scientific requests or durable outputs.

### Scope

- Add one small bundle-level reader that returns the finalized manifest roster when present, otherwise the active authored `cells.tsv` roster.
- Make `experiments/c_study.py` consume that roster and load exactly the nine `*.full` canonical Studies.
- Raise the packed allocation maximum to four and make allocation sizing balanced for arbitrary pending counts up to the selected capacity.
- Update `REMOTE.yaml` to the approved partitions and per-fit CPU/RAM values.
- Update the execution ADR and `docs/FABLE.md` where their stated contracts change.
- Add focused tests for open-roster context preparation, finalized-manifest compatibility, exact missing-parent failure, four-process rendering, and balanced allocation sizes.

### Non-goals

- No incremental C-study-to-HPO or HPO-to-K orchestration.
- No held-out or testing-metric changes.
- No changes to Study/result schemas, `method_index` tolerance, checkpoint format, publication semantics, or feature-ablation closure.
- No scheduler monitoring framework, campaign identity, job migration logic, or remote-output cleanup in product code.
- No speculative validation or abstractions beyond the two immediate consumers: roster lookup and packed launch.

### Protected behavior

- Canonical Study loading and typed request validation remain strict.
- Finalized manifests remain authoritative and manifest-only experiment directories remain unchanged.
- Packed processes remain isolated `srun --exclusive --exact` steps with one GPU each.
- Every successful submission is flushed and synced to `jobs.tsv`; restart skips recorded rows.
- Existing unrelated worktree files and branches remain untouched.

### Required checks

- Focused experiment and execution tests covering changed seams.
- Full test suite.
- `uv run vulture` with manual review of any findings.
- `git diff --check`.
- Independent Standards and Spec review over the fixed baseline-to-head diff; green requires zero actionable findings on both axes.

### Status

- State: ready for implementation
- Implementer: pending
- Reviewer: pending
- Implementation head: pending
- Review result: pending
- Corrections: none

## External activation gate

After Slice 1 is green, re-check the live campaign. Publish only the nine completed full-feature Studies, copy and validate their canonical objects locally, prepare and queue Stage 2, and checkpoint-safely rebalance Stage 1 only when the current scheduler state still shows a material throughput gain. Preserve completed results, checkpoints, logs, Study IDs, and exact request payloads. Update heartbeat authority to count each feature cell satisfied by either its canonical Study or retained hidden result until all 102 are complete.

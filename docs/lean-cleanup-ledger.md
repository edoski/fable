# Lean cleanup implementation ledger

Status: active

## Run constraints

- Initial `main` baseline: `2ea05e0468d6748864d285b8ef5e10c09b5e4ddf`.
- Work directly in the shared checkout. No worktrees or orchestration branches.
- One writer at a time.
- Every implementer and reviewer uses `xhigh` reasoning.
- Each slice has a fresh implementer and a distinct fresh reviewer.
- Review Standards and Spec separately. Green requires zero actionable findings on both axes.
- Do not push, open a pull request, run experiments, submit jobs, alter queued work, or update the
  university checkout.
- After `main` is green, propagate it to `codex/compact-cuda-execution` while retaining the
  device-resident CUDA backing.

## Progress

| Work unit | Baseline | Final head | Implementer | Reviewer | Corrections | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Slice 1 — Typed spine | pending | pending | pending | pending | pending | pending |
| Slice 2 — Modeling and Study | pending | pending | pending | pending | pending | pending |
| Slice 3 — Evaluation and manifests | pending | pending | pending | pending | pending | pending |
| Slice 4 — Operator edge | pending | pending | pending | pending | pending | pending |
| Slice 5 — Python tests | pending | pending | pending | pending | pending | pending |
| Slice 6 — Demo app | pending | pending | pending | pending | pending | pending |
| Slice 7 — Temporal | pending | pending | pending | pending | pending | pending |
| CUDA branch propagation | pending | pending | pending | pending | pending | pending |

## Decisions

- Fresh model defaults support experiment scripts and later Codex-authored typed extensions.
- New HPO, fit, and evaluation work receives a new destination identity. Existing IDs are source
  references. Resumes reuse the complete persisted request.
- Keep destination-based packed-workflow uniqueness because programmatic extensions can retain an
  old destination ID while changing the payload.
- Slice 2 updates accepted ADR 0006 with the approved publication clean break.
- CUDA propagation preserves device-resident historical batching while adopting applicable
  cleanup.

## Slice records

Each completed record will contain the immutable baseline, implementation head, worker identities,
focused checks, review result, correction rounds, proportional integration check, and deferred
checks.

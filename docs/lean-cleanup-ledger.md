# Lean cleanup implementation ledger

Status: active

## Run constraints

- Initial `main` baseline: `2ea05e0468d6748864d285b8ef5e10c09b5e4ddf`.
- Slice 1 worked directly in the shared checkout. The user then authorized temporary worktrees for
  independent slices.
- Each worktree has one writer. Green commits integrate into `main` in plan order.
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
| Slice 1 — Typed spine | `c46e782` | `5912e44` | `/root/slice1_impl` | `/root/slice1_review` | 0 | GREEN |
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

### Slice 1 — Typed spine

- Branch: `main`.
- Implementer commit: `5912e44e7be2d9514d0c8c6f00ce2c056fb5d60b`.
- Changed request construction ownership, deleted `fable.requests`, updated experiment authors,
  and inlined the specified runtime values.
- Implementer checks: focused pytest 30 passed; Ruff passed; Pyright passed; full pytest 128
  passed with 12 environment warnings; diff check passed.
- Reviewer: Standards 0 findings; Spec 0 findings; Vulture passed; `GREEN LIGHT`.
- Orchestrator check: `tests/test_config.py`, 9 passed.
- Deferred: no experiment, Slurm, remote, university, or CUDA execution.

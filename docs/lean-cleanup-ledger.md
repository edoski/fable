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
| Slice 2 — Modeling and Study | `c23cf37` | `11d1639` | `/root/slice2_impl` | `/root/slice2_review` | 0 | GREEN |
| Slice 3 — Evaluation and manifests | `c23cf37` | `26b748f` | `/root/slice3_impl` | `/root/slice3_review` | 0 | GREEN |
| Slice 4 — Operator edge | `c23cf37` | `8e1378d` | `/root/slice4_impl` | `/root/slice4_review` | 0 | GREEN |
| Slice 5 — Python tests | `b36532f` | `d992476` | `/root/slice5_impl` | `/root/slice5_review` | 0 | GREEN |
| Slice 6 — Demo app | `5912e44` | `2fcbd7b` | `/root/slice6_impl` | `/root/slice6_review` | 0 | GREEN |
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

### Slice 2 — Modeling and Study

- Implementation branch commit: `346fb0927c6887f04decdb854eb8ff2b361f0d1a`.
- Integrated `main` commit: `11d1639383923c977fa5027a17ad1c518af1124a`.
- Implementer checks: focused pytest 22 passed; full pytest 128 passed; Ruff, formatting,
  Pyright, and Vulture passed.
- Reviewer: Standards 0 findings; Spec 0 findings; `GREEN LIGHT`.
- Orchestrator check: modeling plus Study tests, 22 passed.
- ADR 0006 and `docs/FABLE.md` now describe direct hardlink publication and the accepted
  leftover-scratch tradeoff.

### Slice 3 — Evaluation and manifests

- Implementation branch commit: `2c2f68ffe8e8a75ce7c5d65089410edf8068afa2`.
- Integrated `main` commit: `26b748f78f96928d5f8b9dd5258893d1ad6e1756`.
- Implementer checks: focused pytest 20 passed; full pytest 125 passed; Ruff, Pyright, and Vulture
  passed.
- Reviewer: Standards 0 findings; Spec 0 findings; `GREEN LIGHT`.
- Orchestrator check: rolling plus manifest tests, 9 passed.
- Integration automatically reconciled its evaluation documentation hunk with Slice 2.

### Slice 4 — Operator edge

- Implementation branch commit: `d8f0bce72198512c717eccafb44cc039e54244fb`.
- Integrated `main` commit: `8e1378dd8ab63b207622cc6d677dc149d3615374`.
- Implementer checks: focused pytest 19 passed; full pytest 128 passed; Ruff, Pyright, Vulture,
  and diff checks passed.
- Reviewer: Standards 0 findings; Spec 0 findings; `GREEN LIGHT`.
- Orchestrator check: execution plus launch tests, 16 passed.
- Combined Slices 2–4 integration gate: full pytest 125 passed; Ruff passed; Pyright passed.

### Slice 5 — Python tests

- Implementation branch commit: `4fb7f7f42a47784400a7d60de07a8e5b9af9fc2a`.
- Integrated `main` commit: `d992476979aa1cd95174015665ef6bd8ae59ac06`.
- Implementer checks: full pytest 124 passed; Ruff, Pyright, and Vulture passed.
- Reviewer: Standards 0 findings; Spec 0 findings; `GREEN LIGHT`.
- Orchestrator check: experiment plus evaluation-resolution tests, 21 passed.
- The suite lost only duplicate fixture detail and call choreography; no migration, shim,
  absence, transition, or architectural-state tests were added.

### Slice 6 — Demo app

- Implementation branch commit: `ae0cacc7f0d3dbb48da0b55aee03255e356a1508`.
- Integrated `main` commit: `2fcbd7b68ef24f7058a65b9debb61de838fe2345`.
- Implementer checks: app tests 32 passed; typecheck and diff check passed.
- Reviewer: Standards 0 findings; Spec 0 findings; `GREEN LIGHT`.
- Orchestrator check: focused App tests, 2 passed.
- Combined Slices 5–6 integration gate: full pytest 124 passed; app tests 32 passed; app
  typecheck passed.

# Lean cleanup implementation ledger

Status: active  
Initial main baseline: `b5fefc5bb96c481a8f5aa57006c0df8cfea6c9e5`  
Initial compact-CUDA baseline: `9693e070b51e5fb250d4bf8c7e4bcbf66eebf3ac`  
Checkout policy: direct sequential commits on `main`, then merge into
`codex/compact-cuda-execution`; no side branch or new worktree  
Protected unrelated path: untracked `docs/experiments/`  
Terminal cleanup: delete this ledger after every authorized slice is green

This ledger is the authoritative specification for the approved cleanup. The goal is less code
and fewer concepts without changing FABLE's scientific meaning, durable record authority,
publication safety, Slurm execution, or mobile inference behavior.

## Fixed decisions

- Use a clean break. Add no compatibility reader, migration path, deprecated alias, transition
  check, or duplicate trusted-value validation.
- Preserve all feature values and order, historical windows and targets, model architecture and
  math, optimization, batching, randomness, artifact contents, evaluation metrics and column
  order, rolling recurrence, experiment cells and order, Slurm resources and script bytes, and
  app-visible behavior.
- Keep `Corpus` ownership validation, `Study` epoch validators, canonical path owners, and
  `src/fable/_runtime.py`.
- Keep HPO's explicit family branches and constructor tuples. Do not replace them with a partial
  or constructor lookup table.
- Keep feature-ablation feature helpers. Do not collapse the feature-unit ownership.
- Keep the compact device-resident historical data design. Do not convert
  `HistoricalDataset` to a dataclass and do not change CUDA transfer or shared-backing semantics.
- Slice 3 contains exactly the three approved standalone Python test reductions. An owning
  implementation slice may delete a test coupled only to private machinery that the same approved
  slice deletes; do not retain a test-only seam or add unrelated test reductions. In particular,
  retain the short-batch weighting assertions in
  `test_epoch_logs_weight_short_batches_in_float64`.
- Skip coincidental app style deduplication.
- Do not push, mutate research storage or jobs, build an image, or change external state.
- Record every branch and worktree created by this run. Remove run-created isolation after safe
  integration and verify the final state. Never delete pre-existing isolation without explicit
  authorization.
- `cells.tsv` is the universal authored experiment design. `jobs.tsv` is the launcher's
  append-only Slurm receipt used for restart skipping; preserve it when launch created it, but do
  not require or invent it at closure.

## Branch and worktree inventory

Pre-run branches were `main`, `codex/compact-cuda-execution`, and the stale local
`codex/lean-cleanup-orchestration`. The user had already approved removing the stale orchestration
branch, and it was deleted before Slice 1.

Pre-run worktrees were the shared checkout plus detached `ed0c` and `f25a` Codex worktrees.
Inspection found:

- `f25a` was clean and held only an obsolete detached planning commit;
- `ed0c` held an uncommitted July 23 app implementation superseded by roughly thirty committed app
  changes on `main`; it lacked the current direct-RPC, ExecuTorch, test, and documentation
  contracts.

The user authorized removing worktrees that were no longer useful. Both detached worktrees were
removed. Their directories are gone and `git worktree list` now contains only the shared
checkout. This run created no branch or worktree.

## GPU and campaign boundary

The authorized changes do not alter corpus construction, features, targets, model computation,
losses, optimizer settings, training order, historical batching, CUDA residency, or runtime
configuration. Existing jobs run inside their already-built immutable image and cannot observe
local commits. These changes alone do not require a replacement `sbuild` image. A new image is
needed only if a future campaign must execute and attest the resulting commit.

## Loop

Each slice starts from an immutable recorded baseline. A fresh implementer follows the
`implement` skill, commits only the slice, and reports its SHA and checks. A distinct read-only
reviewer follows the `code-review` skill against that fixed range, reporting Standards and Spec
separately. Rejected findings return to the same implementer and reviewer until both axes have
zero actionable findings.

Slice 3 has the user's explicit reviewer waiver. It still receives an exact orchestrator diff
check and the full Python suite. No other slice inherits that waiver.

| Slice | Scope | Status | Baseline | Head | Implementer | Reviewer | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Evaluation arrays and result frames | green | `42cb2c4` | `29c31ac2` | `/root/slice1_implement` | `/root/slice1_review` | Standards 0; Spec 0 |
| 2 | Typed experiment-bundle publication | green | `30654b47` | `d98ba42b` | `/root/slice2_implement` | `/root/slice2_review` | 1 correction; Standards 0; Spec 0 |
| 3 | Three Python test reductions | green | `99324dbf` | `3e91e8a1` | `/root/slice3_implement` | waived by user | exact diff; 124 tests passed |
| 4 | Fit-pipeline collapse | green | `61b033fe` | `f5d8a279` | `/root/slice4_implement` | `/root/slice4_review` | Standards 0; Spec 0 |
| 5 | Exact bash template | blocked by 4 | pending | pending | pending | pending | pending |
| 6 | App cleanup | blocked by 5 | pending | pending | pending | pending | pending |
| 7 | Compact-CUDA propagation and parity | blocked by 6 | pending | pending | pending | pending | pending |

## Slice 1 — Evaluation arrays and result frames

### Expected outcome

Evaluation reduction has one direct in-memory representation and substantially less extraction
ceremony, while every published observation and reported scientific value remains identical.

### Scope

- `src/fable/evaluation.py`
- directly affected evaluation tests

Make validated NumPy column mappings the reduction currency. Read the Parquet frame once, retain
the exact ordered `OBSERVATION_SCHEMA` and null checks, then expose its columns as
`dict[str, np.ndarray]` to evaluation, baseline, and rolling reducers.

Change `_economic_metrics` to consume the column mapping plus the selected-policy prefix or an
equivalent direct selector. Remove repeated column extraction from `_reduce`,
`reduce_baselines`, `_reduce_rolling_cell`, and `_rolling_arrays`. Express immediate and deadline
baseline rows through one small iteration.

Delete `_RESULT_SCHEMA`, `_BASELINE_RESULT_SCHEMA`, and `_ROLLING_RESULT_SCHEMA`. Let Polars infer
the learned and baseline result frames. Construct rolling output in the existing explicit
interleaved order:

1. `cell`;
2. one-shot then rolling base-fee savings;
3. one-shot then rolling P50 fee-inclusive savings;
4. one-shot then rolling base-fee optimality gap.

Simplify any tiny finiteness repetition only when the resulting error messages retain their
current cell context.

### Protected behavior

Keep all observation publication fields and types, exact schema/order rejection, null rejection,
request and testing-window identity, all seven learned metrics, both baseline rows and their
three metrics, the four-horizon rolling recurrence, dynamic-origin checks, action bounds,
metric values, error ordering, error context, and public output column order.

### Checks

```text
uv run pytest tests/evaluation
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 2 — Typed experiment-bundle publication

### Expected outcome

Each experiment finishes as one self-contained, typed provenance bundle. Operators can inspect or
reuse its requests, cells, jobs, and record mapping without reconstructing facts from a deleted
scratch directory or a detached summary.

### Scope

- `src/fable/experiments.py`
- `experiments/bundle.py`
- `experiments/feature_ablation.py`
- `experiments/c_study.py`
- `experiments/hpo.py`
- `experiments/k_study.py`
- `experiments/held_out.py`
- directly affected experiment tests

Keep one typed manifest and publish the complete authored bundle instead of transcribing
`cells.tsv` to a detached JSON file and deleting its provenance.

Use one canonical directory:

```text
experiments/<kind>/<experiment_id>/
```

Author under its sibling scratch directory:

```text
experiments/<kind>/.<experiment_id>/
```

On closure, validate every referenced Study, Artifact, or Evaluation through its existing owner,
write one typed manifest inside scratch, and atomically rename scratch to the canonical directory.
The canonical bundle retains its request JSON files, `cells.tsv`, optional `jobs.tsv`, and the
manifest. Reject an existing canonical destination. Existing detached `<experiment_id>.json`
manifests are unsupported.

Represent the manifest as one nonempty ordered cell-to-UUID mapping. Remove `ExperimentEntry` and
the duplicated `experiment_id` field; the canonical directory owns that identity. Loading takes
the requested kind and UUID, reads the manifest inside that exact directory, and returns the
typed mapping.

Deepen `experiments/bundle.py` with the common bundle-opening, request-writing, manifest-writing,
publication, and Study-bundle closure mechanics. Use one consistent request filename format.
Stage modules retain their scientific cell construction, dependency selection, record-owner
choice, reports, and special HPO selected-result output.

### Non-goals

- Do not change HPO model constructors or replace their explicit branches with a lookup table.
- Do not change feature-ablation feature helpers.
- Do not change experiment cells, order, requests, methods, windows, result selection, launch
  packing, or `jobs.tsv`.
- Do not add a legacy manifest reader or migration.

### Checks

```text
uv run pytest tests/test_experiments.py tests/experiments
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

### Review history

Round 1 reviewed `30654b47...ff71e692`.

- Standards: zero actionable findings.
- Spec: one P1 finding. HPO skipped repeated rows before validating every referenced Study;
  K-study and held-out closure accepted path shape rather than loading each Artifact or Evaluation
  through its canonical owner. Empty placeholders therefore passed closure tests.
- Decision: rejected. The same implementer must validate all distinct HPO references, reject
  conflicting mappings, and use canonical Artifact/Evaluation loaders before publication. The
  same reviewer will inspect only the correction range and closure of this finding.

Round 2 reviewed `ff71e692...d98ba42b`.

- Standards: zero actionable findings.
- Spec: zero actionable findings.
- Resolution: HPO loads each distinct Study and rejects conflicting cell mappings; K-study loads
  every Artifact; held-out performs canonical Evaluation loading/reduction; placeholder records
  are rejected. `GREEN LIGHT`.

## Slice 3 — Three Python test reductions

### Expected outcome

The Python suite stays focused on FABLE behavior. Three assertions that only restate framework or
degenerate behavior disappear without reducing scientific, validation, or integration coverage.

### Scope

Delete exactly:

- `tests/test_modeling.py::test_validation_logs_mean_base_fee_cost_over_optimum`;
- `tests/temporal/test_history.py::test_fit_history_supports_shuffled_loading`;
- the one-case parametrization wrapper around
  `tests/test_study.py::test_retained_result_rejects_invalid_epoch_bounds`, leaving one direct
  test with the same assertion.

Change no production file and delete no other test. Remove imports only when these deletions make
them unused.

### Protected behavior

Keep the short-batch float64 and batch-size-weighted optimality-gap coverage, causal-window and
dataset geometry coverage, all `Study` validators, and all other tests.

### Checks

```text
uv run pytest tests/test_modeling.py tests/temporal/test_history.py tests/test_study.py
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

### Execution result

The implementation changed only the three declared test files: two functions were deleted, the
single-case parametrization became one direct assertion, and two unused imports disappeared.
The orchestrator inspected the complete diff, confirmed the weighted short-batch test remains,
and ran the full suite: 124 passed. No dedicated reviewer ran under the user's slice-specific
waiver.

## Slice 4 — Fit-pipeline collapse

### Expected outcome

Fitting reads as one direct preparation-to-result transaction with fewer private relay functions,
while producing the same selected checkpoints, retained results, and durable artifacts.

### Scope

- `src/fable/modeling.py`
- directly affected modeling tests

Delete `_FitOutcome`, `_callbacks`, `_fit_precision`, `_publish_artifact`, and `_fit_candidate`.
Have `_fit` return the selected checkpoint path plus `study.RetainedResult`. Inline the three
callback constructors at the `Trainer` ownership point and inline the precision family ternary
into `Trainer(...)`.

Let `train` directly link the returned best checkpoint to the canonical artifact and then remove
scratch. Let `run_candidate` directly prepare history and the `TrainingDefinition`, call `_fit`,
retain its returned result, and then remove candidate scratch.

Use direct `float(tensor)` conversion where it is exactly equivalent. Remove
`exclude_none=True` from association serialization because the request union has no optional
field.

### Protected behavior

Keep fit seeding, trainer flags, precision by family, callbacks and every callback argument,
resume from `last.ckpt`, best-checkpoint selection, epoch accounting, exact retained objective,
artifact association and checkpoint format, atomic no-clobber hard-link publication, and
failure-preserves-scratch behavior. Do not change models, losses, datasets, optimizer, epochs,
batching, device movement, or `_runtime.py`.

### Checks

```text
uv run pytest tests/test_modeling.py tests/test_study.py tests/cli/test_study.py
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

### Execution result

The five private relay owners and `_FitOutcome` were removed. The test that called only the deleted
private precision helper was removed with that helper; replacing it would have retained a
test-only seam or required a larger synthetic harness. The inline precision rule remains exact.
Focused and full Python checks passed. Independent review found zero Standards and zero Spec
findings and returned `GREEN LIGHT`.

## Slice 5 — Exact bash template

### Expected outcome

The generated Slurm job is readable in source as the bash script it becomes, while the emitted
bytes and external execution behavior remain unchanged.

### Scope

- `src/fable/execution.py`
- only directly necessary execution-test formatting

Replace `_render_allocation_script`'s incremental list construction with one readable
triple-quoted bash template plus a join for the repeated `srun` process blocks.

### Protected behavior

The rendered script must remain byte-for-byte identical for the existing golden. Preserve shell
quoting, heredoc delimiters, packed resource scaling, one exclusive GPU step per process,
slot-specific output, PID collection, wait-all failure propagation, final newline, `REMOTE.yaml`
validation, submission, and job-ID parsing.

### Checks

```text
uv run pytest tests/test_execution.py
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 6 — App cleanup

### Expected outcome

The app has one obvious owner for selection currentness, model-catalog construction, and
action-grouped analytics. User-visible inference, persistence, RPC, and native behavior remain
unchanged.

### Scope

- `app/App.tsx`
- `app/src/inference.ts`
- `app/src/analytics.ts`
- `app/src/model.ts`
- exactly affected app tests

Use the selected `{chain, horizon}` object's identity as the inference revision token and delete
`selectionRevision`. Preserve the active-engine identity check.

Remove `createInferenceEngine`'s overload pair while retaining one production default path and
one explicit dependency-injection path for focused tests. Inline the one-caller inference error
wrapper and result builder where this shortens the flow without weakening causes or safe-integer
checks.

Give analytics one direct grouping-by-selected-action owner and reuse it in the three chart data
builders. Inline the model-catalog factory into the default catalog owner; tests already inject a
`ModelCatalog` interface.

Delete exactly these library-mechanics tests:

- Viem option-object configuration;
- the model-catalog table lookup;
- AsyncStorage JSON round-trip through the test double.

Remove their now-unused fixtures only. Keep app lifecycle/currentness, RPC exact-range and parent
continuity, fee-history alignment, feature parity, ExecuTorch lifetime and output validation,
analytics values, history retryability, and native behavior tests.

### Non-goals

Do not merge semantic style names, change UI copy or layout, alter RPC calls, change inference
math, change model assets/manifests, weaken native tensor validation, or alter persisted run
shape.

### Checks

```text
cd app && npm test
cd app && npm run typecheck
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 7 — Compact-CUDA propagation and parity

### Expected outcome

The reviewed main cleanup reaches the compact-CUDA branch without changing its measured
device-resident execution design. The repository ends with only `main` and the compact-CUDA
branch, one shared checkout, and no temporary orchestration artifact.

### Scope

After every main slice is green, merge `main` into `codex/compact-cuda-execution`. Resolve only
mechanical conflicts required to preserve both the accepted main cleanup and the compact
device-resident historical path. Add no compact-only cleanup.

### Protected behavior

Keep `HistoricalDataset` device-resident batching, one moved backing shared by training and
validation, tensor geometry, dtype, origin arithmetic, evaluation/model device movement, batch
sizes, and `_runtime.configure_torch()` ownership. Do not perform the rejected dataclass/replace
rewrite.

### Checks

```text
uv run pytest tests/temporal/test_history.py tests/test_modeling.py tests/evaluation
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
cd app && npm test
cd app && npm run typecheck
```

No immutable-image build or GPU smoke is required because this run changes no CUDA execution
semantics. Record that those checks were intentionally not run.

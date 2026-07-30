# FABLE residual leanness implementation-review ledger

## Run status

- Phase: execution authorized; ledger freeze pending.
- Product implementation: not started.
- Reviews: not started.
- Planning baseline: `1f1f8873142c54f3cead37682c4b2ed9d8cf85ec`.
- Pre-run branch: `main`.
- Pre-run worktree: `/Users/edo/dev/python/fable`.
- Pre-run status: untracked `docs/experiments/feature_ablation.md`; it is unrelated and protected.
- Run-owned branches/worktrees: none.
- External mutations: none. No push, pull request, scheduler change, or deployment is authorized.
- Checkout policy: approved direct sequential work on `main`, one writer at a time, with no
  worktrees or side branches.

The invoking thread remains the orchestrator. Each slice gets a fresh implementer using the
`implement` skill and, after a fixed committed head exists, a distinct reviewer using the
`code-review` skill. The same implementer and reviewer own any correction rounds. No slice advances
without zero actionable Standards and Spec findings.

## Authority

- Repository instructions: `AGENTS.md`.
- Domain language: `docs/CONTEXT.md`.
- Durable publication: `docs/adr/0006-direct-durable-object-authority.md`.
- Native remote execution: `docs/adr/0007-native-external-execution-boundary.md`.
- Audit source: completed-leanness workflow `wf_8592026e-dae`.
- This ledger owns the corrected scope and accepted tradeoffs. Audit proposals not recorded as
  approved here are not implementation authority.

## Accumulated decisions

### Approved

| Decision | Required shape |
| --- | --- |
| Derive observation fee columns after inference | Keep observation bytes and schema exact. Retain bounded preallocation, one flat canonical outcome range, the predicted-log finite check, and all origin/action arithmetic. This is a clarity change; no performance claim is authorized. |
| Inline `_load_observations` | Keep strict request identity, canonical schema, nonnull data, and exact ordered testing-window coverage. Keep `_read_observations`, which is shared by ordinary and rolling loads. |
| Share finite-metric validation | One small private float-metric guard may replace the three repeated checks only if callers become shorter and the existing subject-specific errors remain. |
| Remove the `publish_study` canonical precheck | Keep first-result reuse and `os.link` as the sole atomic no-overwrite gate. Slightly later failure after small local validation is accepted. |
| Uniform per-slot Slurm logs | Every allocation size, including one process, writes `<job_id>-<slot>.out`; `%j.out` remains the allocation log. |
| Publish manifest-only experiment directories | Canonical experiment directories retain only `manifest.json`; remove `_repoint_requests` and temporary requests/TSV files without weakening create-only publication. |
| Share typed prepare-row writers | Tune, Train, and Evaluate helpers each own request numbering, fixed TSV schema, durable-ID derivation, and Tune method expansion. Do not add a stringly column parameter or caller-supplied redundant UUID. |
| Share held-out report printing | One private printer owns manifest traversal, cell-column construction, concatenation, and TSV output; `report` and `baselines` retain their distinct reducers. |
| Use one canonical feature-unit roster | Preserve exact unit and feature order, including Ethereum-only `exact_forming_base_fee` in its current third position. |
| Inline `findBrokenLink` | Keep the parent-link check, ordering, and exact failure message in `sync`. |
| Centralize feature-route predicates | Feature semantics own the predecessor and priority-fee predicates; RPC and feature construction do not re-derive them independently. |
| Validate priority-fee presence at both owning seams | RPC validates raw `eth_feeHistory` reward presence and exact row count. Feature construction retains one direct-input invariant and removes repeated per-feature branches. |
| Replace the small argmax loop | Use the one-expression first-maximum implementation only while logits remain finite and fixed to length 2–5. |
| Consolidate App selection state/update logic | One selection-shaped React state and one identity ref may replace separate chain/horizon state and duplicate setters. Preserve stale-result rejection for horizon ABA and engine replacement. |
| Share overlay scaffolding | One narrow overlay owns `Modal`, root, dismissible backdrop, and close semantics for the three existing consumers. Screen-specific panels remain screen-owned. |
| Keep the two network selectors separate | The setup grid and compact Analytics picker retain their distinct presentation. No accessibility-only cleanup is in scope for this demo. |
| Perform bounded chart cleanup | Keep the three named semantic charts and negative-axis behavior. Remove only shallow scale-prop plumbing, move the fee legend to its owner, and require visual verification. Do not add one broad rest-prop chart abstraction. |
| Replace chart builders with `waitBuckets` | One reduction computes per-offset run count, realized savings, and fee means. Preserve pending-outcome semantics, Gwei conversion, empty states, and chart values. |
| Narrow storage-error reporting | Keep initial load/corrupt-storage diagnostics. Remove duplicate save-failure banner state because the inference modal owns save failure. |
| Simplify history aliases and outcome assignment | Remove the two type aliases and redundant outcome spread. Keep the composite persisted run ID. |
| Resolve pending outcomes independently | One RPC failure leaves only that run pending; successful sibling outcomes are committed and failed runs remain retryable. |
| Retain duplicate allocation guards | UUID generation is trusted, but replayed explicit request IDs and duplicate persisted bundle rows remain rejected before concurrent work can share scratch or canonical ownership. |

### Protected non-goals

- Do not flatten `RetainedResult` or remove selected/completed epoch provenance.
- Do not merge workflow and candidate submission records.
- Do not remove allocation size, duplicate durable-identity, or duplicate Study-slot guards.
- Do not remove local method-index validation.
- Do not loosen `ExperimentManifest` key/map constraints or replace its named strict schema with a
  loose adapter.
- Do not remove HPO cell consistency or launch journal identity checks.
- Do not weaken mobile-export scratch publication.
- Do not remove `safeBigInt`, model-runtime disposal/load-failure cleanup, zero-base-fee numerical
  protection, or the App selection identity token.
- Do not deduplicate semantic styles by replacing them with generic utility names.
- Do not change Analytics horizon initialization.

## Decision gates

None.

## Ordered slices

Every slice baseline is re-recorded immediately before dispatch. Slice status advances only through
its implement, fixed-range review, and correction gates.

### Slice 1 — Evaluation loading and finite reductions

Status: pending.

Scope:

- Derive the seven economic observation columns from one flat canonical outcome range after model
  inference rather than assembling them per batch.
- Inline `_load_observations` into `_load_evaluation`.
- Introduce the approved finite float-metric guard only if the committed implementation reduces
  total branching and preserves error subjects.

Non-goals:

- Observation schema, window, rolling-origin, action-range, or numerical-contract changes.
- Performance or memory improvement claims.

Expected outcome:

The inference loop owns only model outputs, economic observation columns are derived once from
canonical flat arrays, evaluation loading has one fewer private hop, and all three reductions
express the same finite metric rule once without changing any observation or metric.

Checks:

- `uv run pytest tests/evaluation`
- `uv run ruff check src/fable/evaluation.py tests/evaluation`
- `uv run pyright`
- `uv run vulture`, with manual review of every reported finding.

Dependencies and gates: none.

### Slice 2 — Uniform Slurm process logs

Status: pending.

Scope:

- Remove the single-process output special case.
- Update the golden execution test and remote-submission documentation.

Non-goals:

- Submission-record unification.
- Allocation-count, duplicate-identity, GRES, method-index, payload, or immutable-image changes.

Expected outcome:

Operators find every process log at the same `<job_id>-<slot>.out` address regardless of allocation
size, while `%j.out` remains the allocation-level log.

Checks:

- `uv run pytest tests/test_execution.py tests/cli/test_study.py`
- `uv run ruff check src/fable/execution.py tests/test_execution.py`
- `uv run pyright`

Dependencies and gates: Slice 1 green.

### Slice 3 — Canonical publication and experiment mechanics

Status: pending.

Scope:

- Remove the redundant `publish_study` canonical existence precheck while retaining first-result
  reuse and atomic `os.link` publication.
- Publish canonical experiment directories containing only `manifest.json`.
- Delete `_repoint_requests`.
- Add typed Tune, Train, and Evaluate cell writers that own request paths, fixed headers, durable
  IDs, and Tune method expansion.
- Share held-out TSV printing.
- Correct `docs/FABLE.md` from stale `<UUID>.json`/`entries` prose to the implemented
  `<UUID>/manifest.json` flat mapping and manifest-only closure.

Non-goals:

- Manifest schema loosening.
- HPO, launch-journal, or canonical-record verification removal.

Expected outcome:

Study and experiment publication retain their atomic no-overwrite authority, experiment authors no
longer repeat row-schema mechanics, closed experiments retain only their canonical reference
manifest, and both held-out reports use one output path without changing bytes.

Checks:

- `uv run pytest tests/test_study.py tests/test_experiments.py tests/experiments`
- `uv run ruff check src/fable/study.py src/fable/experiments.py experiments tests/experiments`
- `uv run pyright`
- Exact inspection of canonical directory contents after each experiment closure.

Dependencies and gates: Slice 2 green.

### Slice 4 — Feature-ablation roster

Status: pending.

Scope:

- Replace staged mutable unit construction with one canonical constant and one Ethereum filter.
- Replace `_flatten_units` with the direct ordered expression if clearer.

Non-goals:

- Cell count, labels, ordering, methods, windows, or scientific selection changes.

Expected outcome:

The frozen feature-ablation design reads from one ordered roster, with chain-specific exclusion
visible in one place and byte-identical authored cells.

Checks:

- `uv run pytest tests/experiments/test_feature_ablation.py`
- `uv run ruff check experiments/feature_ablation.py tests/experiments/test_feature_ablation.py`
- `uv run pyright`
- Assert the existing 102-cell order and labels remain exact.

Dependencies and gates: Slice 3 green.

### Slice 5 — App feature-route and inference mechanics

Status: pending.

Scope:

- Centralize predecessor and priority-fee route predicates.
- Validate raw priority-fee reward presence and exact length in RPC.
- Retain one direct feature-input invariant while removing repeated P50/P90 branches.
- Inline the one-use broken-link search into RPC synchronization.
- Replace the finite 2–5-element argmax loop with the approved first-maximum expression.

Non-goals:

- `safeBigInt`, runtime lifecycle, RPC timeout/retry, model I/O, or feature arithmetic changes.

Expected outcome:

One feature owner decides route requirements, each input seam rejects malformed priority-fee data
once, RPC synchronization stays direct, and prediction decoding is smaller while producing
identical selections and errors.

Checks:

- `npm test -- test/features.test.ts test/rpc.test.ts test/inference.test.ts` from `app/`.
- `npm run typecheck` from `app/`.
- `npm test -- --run` from `app/`.

Dependencies and gates: Slice 4 green.

### Slice 6 — App selection and overlay ownership

Status: pending.

Scope:

- Consolidate chain/horizon React state and update logic around one selection value.
- Extract one narrow shared overlay scaffold.

Non-goals:

- Removing the selection identity ref or stale-result tests.
- Chart or network-grid changes, including accessibility-only cleanup.
- Screen-specific dialog contents, network styling, or Analytics horizon behavior.

Expected outcome:

Selection changes update one coherent value while preserving both stale-run gates, and all existing
dialogs share identical modal/backdrop dismissal behavior without losing screen-specific content.

Checks:

- Focused App and screen tests, including horizon change-and-return and engine replacement.
- `npm run typecheck` from `app/`.
- `npm test -- --run` from `app/`.
- Visual inspection of all three overlays.

Dependencies and gates: Slice 5 green.

### Slice 7 — Analytics buckets and partial outcome progress

Status: pending.

Scope:

- Remove shallow chart scale-prop plumbing and move the fee legend to the fee chart while retaining
  the three named charts.
- Replace three chart-data builders and their private grouping pass with `waitBuckets`.
- Keep initial storage-load diagnostics while removing duplicate save-failure banner state.
- Remove the two history type aliases and redundant outcome spread while retaining the composite
  run ID.
- Resolve each pending run independently so successful siblings commit when another RPC read fails.

Non-goals:

- Zero-base-fee, pending-outcome, history ordering, selection filtering, or chart-format changes.
- Broad rest-prop chart abstraction or chart identity changes.

Expected outcome:

Analytics computes each wait bucket once through three visibly unchanged named charts, storage
errors have one owner per failure path, history types and assignment stay direct, and outcome
refreshes retain all successful progress while leaving failed runs unchanged and retryable.

Checks:

- Focused analytics, history, and App tests including mixed successful/failed outcome resolution.
- `npm run typecheck` from `app/`.
- `npm test -- --run` from `app/`.
- Visual verification of all three charts, including negative savings and the fee legend.
- Final repository integration: `uv run pytest`, `uv run vulture`, app typecheck, app full tests,
  `git diff --check`, protected-path status, and ledger reconciliation.

Dependencies and gates: Slice 6 green.

## Execution log

No workers have been dispatched. No implementation or review commits exist.

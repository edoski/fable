# FABLE architectural cleanup ledger

Status: execution authorized; product implementation has not started.

This file is the authoritative local specification for the cleanup run. The orchestrating thread
owns it. Implementers and reviewers must not edit it.

## Run policy

- Use the implementation-review loop. Each slice gets a fresh implementer using the `implement`
  skill, followed by a distinct fresh reviewer using the `code-review` skill over immutable
  baseline and head commits. Standards and Spec are reviewed separately. A slice advances only
  after both axes report zero actionable findings.
- Return rejected findings to the same implementer. Send correction commits to the same reviewer
  and review only the correction range against the outstanding findings.
- Work directly on `main` for Slices 1–3, then reconcile onto the existing
  `codex/compact-cuda-execution` branch in the shared checkout, with one writer at a time. Create no
  branch or worktree. Do not push, open a pull request, mutate GitHub or remote systems, submit
  Slurm jobs, build research images, or run live RPC without separate authorization.
- Use a clean break. Add no compatibility path, migration, registry, generalized lifecycle layer,
  architectural transition test, or defensive validation of trusted internal values.
- Preserve unrelated work. The pre-existing untracked `docs/experiments/` directory is user-owned
  and outside this run unless the user explicitly changes that decision.
- Keep this ledger temporary. Delete it only after every authorized slice is green and final
  integration checks pass.

## Pre-run state

- Recorded: 2026-08-01, Europe/Rome.
- Repository: `/Users/edo/dev/python/fable`.
- Branch: `main`.
- Immutable planning baseline: `2df783592ca048c72960261cbbc45cfe4860da88`.
- Existing reconciliation target: `codex/compact-cuda-execution` at
  `bc9532cfecba0d908188c8d234b5666c716701ed`; it contains the planning baseline and is 22 commits
  ahead with its approved compact-CUDA/data-loader delta.
- Existing worktrees: only `/Users/edo/dev/python/fable`, checked out on `main`.
- Pre-existing unrelated status: `?? docs/experiments/`.
- Run-owned branches/worktrees: none.
- Planning verification already completed:
  - `uv run pytest -q -p no:cacheprovider`: 89 passed;
  - `uv run ruff check --no-cache .`: passed;
  - `uv run ruff format --check --no-cache .`: 47 files formatted;
  - `uv run pyright`: zero errors and warnings;
  - `PYTHONDONTWRITEBYTECODE=1 uv run vulture`: no findings;
  - from `app/`, `npm test -- --run --no-cache`: 35 passed across 7 files;
  - from `app/`, `npm run typecheck`: passed;
  - canonical isolated mobile-export suite: 11 passed.

Before every slice, replace its proposed baseline with the actual immutable repository head and
record exact status. Only the ledger and the known unrelated `docs/experiments/` directory may be
present before product implementation begins.

## Governing sources

- `AGENTS.md`.
- `docs/CONTEXT.md`.
- `docs/FABLE.md`.
- `docs/adr/0006-direct-durable-object-authority.md`.
- `docs/adr/0007-native-external-execution-boundary.md`.
- `docs/agents/issue-tracker.md`.
- The decisions below. Finder audits are evidence, not authority.

## Decisions already aligned

### App consistency and analytics

1. Restore ordered selection and history publication in `App.tsx`.
   - One App-owned selection record distinguishes the last applied selection from the latest user
     intent.
   - Selection application shares the existing rejection-safe history FIFO. Repeated requests
     coalesce to the latest intent, including a change away and back while a save is pending.
   - A history update accepted for the current engine/selection remains ordered through durable save
     and in-memory publication. Selection cannot become applied halfway through that transition.
   - Preserve stale-engine and stale-selection rejection, one durable write for an accepted update,
     fresh engine construction only when the finally applied chain changes, and retryable pending
     outcome resolution.
   - Replace the current horizon-away-and-back test with one blocked-save ordering test covering a
     chain/horizon change and reversion. Keep the separate replaced-engine stale-result test.

2. Stop rendering unavailable savings buckets as measured zero.
   - `SavingsByWaitChart` passes only buckets with non-null `savingsPercent` to the chart, matching
     `BaseFeeByWaitChart`.
   - Keep all buckets in the recommendation-count chart. Keep the existing empty-outcomes state and
     analytics null semantics.
   - Preserve the current `winPercent` formula, field name, and visible "Win rate" label. It remains
     the percentage of resolved wait recommendations with strictly positive savings; act-now and
     pending runs remain outside its denominator.
   - Do not add snapshots or style tests.

3. Compose inference cleanup directly.
   - `InferenceEngine.dispose()` attempts model disposal even if session unwatching throws, using a
     compact `try/finally` rather than a lifecycle abstraction.
   - App cleanup contains a rejected disposal promise so unmount/replacement cannot create an
     unhandled rejection.
   - Preserve idempotent model disposal, immediate execute-after-dispose rejection, queued native
     execution, and model deletion order.

### Scientific experiment authority

4. Make the selected context Study the owner of HPO's control Method.
   - `_methods` accepts the source `request.methods[0]` for each selected chain/family Study.
   - Capacity zero derives from that exact upstream model and fit. HPO overrides only the dimensions
     it intentionally explores: dropout, learning rate, and weight decay.
   - One Transformer-capacity table owns `model_width`, `attention_heads`,
     `transformer_layers`, `feedforward_width`, and `head_hidden`. Transformer-LSTM reuses those
     dimensions and adds only `lstm_hidden=model_width` and `lstm_layers=1`.
   - Preserve the nine Latin-hypercube rows, their order, family, context winner selection and tie
     behavior, printed output, authored cell order, and durable Study provenance.

5. Derive held-out support from the loaded K-study roster.
   - Parse the K-study manifest before constructing testing windows and derive the largest authored
     horizon from its cells.
   - Use that value for complete-outcome separation from validation and corpus-tail support.
   - Keep the explicit `K=2,3,4,5` rolling-comparison policy; it is separate scientific meaning.
   - Reject malformed or empty horizon rosters through the existing typed/canonical owners rather
     than adding a second general validation framework.

### Verified internal ceremony removal

6. Remove validation options already owned by their schemas or parent command.
   - Remove per-call `strict=True` from the CLI workflow parser, canonical Study loader, and private
     candidate-result loader. `StrictFrozenRecord` remains the strict owner and coercive JSON must
     still be rejected.
   - Remove `add_completion=False` from the two nested Typer applications. The root application
     remains the completion owner and nested help must remain unchanged.
   - Remove `Field(min_length=1)` from `Study.trials`. Nonempty `TuneRequest.methods` plus exact
     trial/Method cardinality continues to make an empty Study invalid.

7. Apply approved focused test cleanup without weakening FABLE-owned behavior.
   - In the native-loss fixture, keep `backward()` and assert that both output heads receive a
     gradient, but remove exact derivative matrices that retest PyTorch autograd.
   - Remove the evaluation observation null-count assertion; the exact row fixture immediately
     below already proves that every value is present.
   - Delete `test_request_defaults_mint_and_persist_destination_identity` and its unused `json`
     import. Generated destination identities and strict serialized dispatch remain covered through
     CLI and experiment interfaces; the deleted assertions otherwise retest `uuid4`, `Literal`, and
     Pydantic serialization.
   - Delete `test_block_frame_owns_one_valid_canonical_frame` and its now-unused Polars assertion
     import. Exact range, schema rejection, and input/returned-frame isolation retain the actual
     `BlockFrame` contract.
   - Keep the existing 102-cell launcher integration test. A replacement would add fixture code
     while losing real feature-ablation authorship, packed-allocation, journal, and resume coverage.
   - Keep all raw-input, scientific/numerical, causal, durable-publication, native-runtime, resume,
     and demonstrated-race tests.

### Additional verified simplifications from the attached audit

8. Simplify current implementation details that have one exact behavior.
   - In `_TransformerModel`, branch once on `TransformerLstmDefinition` to assign both the optional
     LSTM and its output width. Do not derive model-family scientific configuration here.
   - Return `np.column_stack(columns)` directly from `_raw_feature_rows`; every feature branch is
     Float64 and `column_stack` already creates the required C-contiguous matrix. Keep downstream
     finite Float32 transformation.
   - Replace HPO's `setdefault` row loop with one insertion-ordered dict comprehension. Authored rows
     repeat one Study ID per cell; do not add corrupt-bundle reconciliation machinery.
   - Make `HistoricalDataset.__init__` accept its backing, Experiment, window, and fitted target
     state; let it compute and retain origin rows, labels, and standardized targets directly. Delete
     `_build_dataset`. Deliberately retain the duplicate training-outcome pass: this choice removes
     roughly 10–13 source lines and avoids an optional compute-once construction path.
   - Inline the sole training use of `_FitModule._loss`. Validation continues to compute the same
     loss from its already-needed model output.
   - Make `BlockFrame.select_range` return through the public constructor instead of bypassing it
     with `object.__new__` and manual field assignment. Preserve inclusive slicing and isolation.
   - Remove the explicit CUDA matmul `allow_tf32` assignment and its redundant assertion. Pinned
     Torch 2.7.1 makes `set_float32_matmul_precision("high")` the owner of that flag; keep the
     independent cuDNN TF32 assignment and assertion.
   - Remove `method_index` from `_CandidateResult`, delete its filename comparison and focused test,
     and stop writing it in new scratch results. Hydrate active-campaign scratch JSON with
     `extra="ignore"` so the old transport-only field is discarded at the private boundary while
     strict request/result value types remain enforced. Canonical Study data remains unchanged.

9. Simplify the fixed demo presentation without creating new modules.
   - Remove every app-owned `accessibility*` and `accessible` prop. In particular, make
     `AppHeader`'s status table own colors only, remove its accessibility-only chain/label data, and
     render the same visible dot directly.
   - Remove `Overlay.backdropLabel` and its three caller arguments. Keep backdrop presses, modal
     close behavior, disabled interaction, hit slop, and all visible presentation unchanged.
   - Make `InferenceScreen` own the one shared `ScrollView`, page style, and `Inference` title.
     `Setup` and `Result` own only their branch content; the error overlay stays outside the scroll
     view. Preserve layout and state behavior.
   - Reuse the existing `GWEI` constant in `formatGwei`.
   - Rename the Analytics screen's initialization-only prop to `initialHorizon`.
   - Remove the unused exported `RunSummary` and one-implementation `ModelCatalog` type declarations;
     rely on inferred return shapes without adding replacement interfaces.

10. Stream mobile program publication.
    - Open the destination in binary-write mode and call ExecuTorch's `program.write_to_file` rather
      than materializing `program.buffer` as one contiguous copy.
    - Preserve XNNPACK inspection, the exact written program, subsequent host loading, two-sample
      eager parity, finiteness, selected action, decoded fee, and atomic final publication.

11. Centralize positive live base-fee eligibility at the RPC seam.
    - `blockRow` rejects a missing or nonpositive `baseFeePerGas` once at the raw external adapter.
      The three supported EIP-1559-style routes have positive protocol base fees.
    - Downstream inference and app-owned history trust resolved outcomes. Delete `validOutcome`, both
      `Unavailable` presentation branches, and synthetic zero-baseline analytics cases.
    - A rejected raw outcome remains unresolved and retryable through the existing history policy.
      Pending outcomes remain excluded from realized savings and wait-bucket fee means.
    - Keep one compact RPC case for null/nonpositive base fees; add no stored-history migration or
      schema validator.

12. Trust experiment-authored allocation identities.
    - `submit_workflows` and `submit_candidates` pass their typed inputs directly to
      `_submit_allocation`. Delete `_workflow_identity`, both within-allocation destination
      uniqueness checks, and their two focused tests; do not replace them with whole-request
      equality.
    - Supported experiment authors mint each workflow destination once, expand each TuneRequest into
      unique method indices, and preserve row identity on resume. A collision requires malformed
      manually edited input, which is outside the supported experiment/Codex path.
    - Keep typed request hydration, candidate method-range validation, one-to-three input
      cardinality, GRES scaling, raw remote configuration checks, durable job journaling, and
      ordered payload submission.

## Pending decisions

None.

## Protected behavior and rejected directions

- Keep exact feature formulas, causal context/outcome geometry, target/loss/decode semantics,
  validation economics, rolling-comparison semantics, deterministic experiment ordering, and
  metric denominators unless a decision explicitly changes one.
- Keep canonical durable schemas and typed associations, UUID/object identity checks, canonical
  paths, atomic no-clobber publication, owner-local scratch, and candidate-specific full-state
  resume. The private active-campaign candidate loader may discard its retired transport-only index;
  this does not change canonical Study data.
- Keep RPC cardinality/type/sign/parent-continuity checks, exact bigint fee arithmetic, finite
  float32 features, native tensor validation, exporter parity, and XNNPACK proof.
- Keep `TuneRequest.method_at`, Slurm job journaling, and fail-before-publication behavior.
- Do not restore a custom gradient-clipping hook or add tests for it. Lightning owns the configured
  clipping behavior; current validation retains complete-metric finiteness rejection.
- Do not add a process-global ExecuTorch queue without demonstrated separate-runtime native failure.
- Do not merge scientific feature construction into the RPC adapter, split the fixed demo's style
  module, introduce generated cross-language contracts, or add generalized ports/registries.
- Do not delete `tests/test_experiments.py`: its canonical loader/address and empty-manifest coverage
  is not supplied by the feature-ablation publication test.
- Keep `evaluation_json_path`. Although production publication writes within its owner-local scratch
  directory, the helper is the canonical filename owner parallel to the other ADR-0006 addresses;
  deleting it would move that knowledge into its test for at most three lines of reduction.
- Do not generalize `write_train_cells` and `write_evaluate_cells` through a dynamic record-column
  interface merely to remove a few repeated lines.
- Do not replace the exporter's YAML-to-JSON strict-validation path with `validate_python(strict=True)`;
  strict Python validation rejects YAML UUID strings while strict JSON validation accepts their
  canonical representation.
- Do not alter, delete, or commit `docs/experiments/`; it contains the user's live feature-ablation
  campaign note and remains untracked. Do not alter `outputs/`. `outputs/` is already protected
  locally by `.git/info/exclude`; repository ownership of that local directory is outside this
  cleanup. Dynamic campaign-status prose is outside the architectural cleanup.
- Do not treat the attached audit's unnamed "six small app items" as scope. The concrete surviving
  micro-cleanups found in current code are recorded in decision 9; several earlier candidates had
  already landed before this baseline.

## Slice 1 — Core data, training, and trusted internal seams

Status: green.

Baseline: `ac60331ac3f790e4827ae36991d34ac593ff8ab6`.

### Expected outcome

Core preparation, modeling, request hydration, and their tests contain no shallow internal seam or
duplicated trusted-value option, while scientific/numerical behavior and durable identities remain
unchanged.

### Scope

- Implement decision 6 and the applicable parts of decision 7.
- Implement decision 8.
- Implement decision 12 and delete only its two focused duplicate-identity tests.
- Update tracked runtime prose to make clear that `high` matrix-multiplication precision owns CUDA
  matmul TF32 while the separate cuDNN flag remains.
- Update imports and focused tests only where the final interface requires it.

### Focused checks

- `uv run pytest -q -p no:cacheprovider tests/corpus tests/temporal tests/test_modeling.py
  tests/test_min_block_fee.py tests/test_study.py tests/cli tests/evaluation/test_evaluate.py
  tests/experiments/test_launch.py`.
- Targeted JSON probes proving workflow and Study strictness plus active-campaign candidate hydration:
  the retired extra index is ignored, while coercive request/result values remain rejected.
- Root Ruff check/format, Pyright, Vulture with manual finding review, full Python tests,
  `git diff --check`, and exact status audit.

### Non-goals and gates

- No model/loss/target/feature semantic change, GPU training, Slurm activity, remote checkout, or
  research image build.
- No app or mobile-export change.

### Implementation-review record

- Implementer: `/root/slice1_implement` using the `implement` skill.
- Implementation head: `35ee2fdf3d1b97cb23341609a6265865aa1d0782`
  (`refactor(core): simplify trusted internal seams`); 16 files, 62 insertions, 181 deletions.
- Implementer checks: focused suite 57 passed; active-campaign/strict JSON and Torch 2.7.1 TF32
  probes passed; Ruff check and format check passed; Pyright zero errors/warnings; Vulture no
  findings; full Python suite 84 passed; diff and status audits passed.
- Reviewer: `/root/slice1_review` using the `code-review` skill, with independent parallel
  Standards and Spec axes over the fixed `ac60331...35ee2fd` range.
- Standards: zero actionable findings and zero smell findings. The user-approved one-call
  `extra="ignore"` scratch-envelope exception is bounded and documented; nested scalar strictness
  remains intact.
- Spec: zero actionable findings. Decisions 6–8 and 12 are complete; protected scientific,
  publication, resume, allocation, and cuDNN TF32 behavior remains intact.
- Correction rounds: none.
- Final result: `GREEN LIGHT`. Reviewer mutation audit found no changes; only protected
  `?? docs/experiments/` remained.
- Intentionally unrun: GPU training, Slurm/SSH, live RPC, remote checkout/image build.

## Slice 2 — Experiment authorship and durable grouping

Status: green.

Baseline: `97f417b723cc373f3429a0b75d17f5638f31f9ae`.

### Expected outcome

HPO and held-out authoring derive shared scientific facts from their canonical upstream records or
one explicit local owner, so later code edits cannot silently desynchronize experiment stages.

### Scope

- Implement decisions 4 and 5.
- Share the approved Transformer/Transformer-LSTM HPO capacity table without changing any authored
  Method value or row order.
- Simplify HPO manifest row deduplication without changing first-cell insertion order, selected
  Study identity, printed selection order, or fail-before-publication.
- Correct tracked FABLE prose only where metric or experiment terminology becomes inaccurate.
- Do not rewrite dynamic campaign-status prose or touch the untracked campaign note.

### Focused checks

- `uv run pytest -q -p no:cacheprovider tests/experiments tests/test_experiments.py
  tests/test_study.py tests/evaluation`.
- Explicit assertions for upstream control-Method inheritance, the nine HPO rows, shared capacity
  dimensions if approved, manifest order, and held-out bounds derived from the authored K roster.
- Root Ruff check/format, Pyright, Vulture with manual finding review, full Python tests,
  documentation residue search, `git diff --check`, and exact status audit.

### Non-goals and gates

- Do not change completed experiment records, current Slurm jobs, feature-ablation conclusions,
  selection objectives, or rolling policy.
- No new experiment, training, evaluation, Slurm action, or remote mutation.

### Implementation-review record

- Implementer: `/root/slice2_implement` using the `implement` skill.
- Implementation head: `fd1e8a52efb32b5e3460068806aecd60ac856f85`
  (`refactor(experiments): derive authored scientific controls`); five files, 166 insertions,
  92 deletions.
- Implementer checks: focused suite 36 passed; Ruff check and format check passed; Pyright zero
  errors/warnings; Vulture no findings; full Python suite 84 passed; documentation residue, diff,
  and exact status audits passed.
- Reviewer: `/root/slice2_review` using the `code-review` skill, with independent parallel Standards
  and Spec axes over fixed `97f417b...fd1e8a5`.
- Standards: zero actionable findings and no baseline smells. Exact L9 order/values, shared
  capacities, tie behavior, held-out geometry, documentation, and ADR boundaries verified.
- Spec: zero actionable findings. Decisions 4–5 and the HPO comprehension cleanup are complete;
  upstream Method inheritance, selected Study identity, maximum-horizon derivation, and fixed
  `K=2…5` rolling policy remain exact.
- Correction rounds: none.
- Final result: `GREEN LIGHT`. Reviewer made no mutations; only protected
  `?? docs/experiments/` remained.
- Intentionally unrun: authored experiments, GPU training, Slurm/SSH, live RPC, remote checkout and
  image build, app/mobile checks.

## Slice 3 — Demo consistency and mobile publication

Status: green.

Baseline: `c2c34dfe8b997a1fb183b08a70e10095521a497f`.

### Expected outcome

The demo applies selection, persistence, analytics, native cleanup, and model publication through
direct owners; unavailable observations are never presented as measured values; and UI wording
states the exact metric being shown.

### Scope

- Implement decisions 1–3.
- Implement decisions 9–11.
- Update the tracked app lifecycle description so it names the ordered applied/intended selection
  and history owner rather than claiming one immediate identity gate protects history commits.

### Focused checks

- App TDD at the `App` interface for a blocked history save plus selection change/reversion.
- Existing analytics semantic tests and direct inspection of the filtered private chart mapping; do
  not extract a test-only chart interface or add a snapshot suite.
- Inference/model disposal tests proving both cleanup attempts, contained rejection, idempotence,
  queued execution, and deletion order.
- From `app/`, full Vitest and TypeScript checks.
- Canonical isolated mobile-export 11-test suite, including real XNNPACK export and host execution.
- Root `git diff --check` and exact status audit.

### Non-goals and gates

- No generated model assets, manifest change, live RPC, dependency update, native app build,
  simulator/device acceptance, or visual-parity claim without a runnable development build.

### Implementation-review record

- Implementer: `/root/slice3_implement` using the `implement` skill and App-interface TDD.
- Initial implementation head: `360159113769654f1bc3820b4283fece22cbdee5`
  (`refactor(app): order demo state and publication`); 18 files, 212 insertions, 257 deletions.
- Implementer checks: required App seam failed before implementation and passed afterward; full
  Vitest 36 passed; TypeScript passed; canonical mobile-export suite 11 passed including real
  XNNPACK lowering/host execution; root Python suite 84 passed; Ruff check/format, Pyright, Vulture,
  diff, and status audits passed.
- Reviewer: `/root/slice3_review` using the `code-review` skill, with independent parallel Standards
  and Spec axes over fixed `c2c34df...3601591`.
- Initial review: Spec zero findings. Standards rejected the head with one P3 Mysterious Name:
  `serializeHistory` had become the owner of both history and selection ordering.
- Correction round 1: the same implementer committed
  `a50c4887638783daeff3edf48e5d6169897e6196`
  (`refactor(app): name ordered update queue`), renaming all five references to
  `enqueueOrderedUpdate`. Focused App tests 4 passed; full Vitest 36 passed; TypeScript and diff
  checks passed.
- Correction review: the same reviewer inspected only fixed `3601591...a50c488`; Standards and Spec
  both returned zero findings. The correction was rename-only and closed the sole finding.
- Final head/result: `a50c4887638783daeff3edf48e5d6169897e6196`; `GREEN LIGHT`.
- Reviewer mutation audits found no changes; only protected `?? docs/experiments/` remained.
- Intentionally unrun: live RPC, generated model assets, native app build, simulator/device and
  visual acceptance, GPU training, Slurm/SSH, remote checkout, research image build.

## Main integration record

- Product head: `ad75d5edcc4cb9e36d769d38e688d29224432ef1`.
- Root Python suite: 84 passed with only expected local MPS/Lightning warnings.
- Ruff check passed; Ruff format check found 47 files already formatted; Pyright reported zero
  errors/warnings; Vulture reported no findings.
- App Vitest: seven files and 36 tests passed; TypeScript typecheck passed.
- Canonical isolated mobile-export suite: 11 passed, including portable negative control, real
  XNNPACK lowering, serialization, and host execution; dependency deprecation/experimental warnings
  only.
- Documentation/identifier residue search found removed names only in this historical ledger. Diff,
  branch, worktree, and status audits passed; only protected `?? docs/experiments/` remained.
- Intentionally unrun: GPU training, Slurm/SSH, live RPC, remote checkout/image build, generated
  model assets, native app build, simulator/device and visual acceptance.

## Slice 4 — Compact-CUDA branch reconciliation

Status: green.

Compact baseline: `bc9532cfecba0d908188c8d234b5666c716701ed`.

Finalized main parent: `44b309ee44c5a6f8a4ed00a4e6f30c425de79988`.

### Expected outcome

The existing compact-CUDA branch contains the complete reviewed cleanup from `main` while retaining
only its approved data-loader and compact-CUDA execution differences. Shared modules, tests, and
documentation express the same cleanup and scientific contracts on both branches.

### Scope

- Switch the sole shared checkout from `main` to the existing `codex/compact-cuda-execution` branch
  only after main is clean and all prior workers are idle.
- Merge the finalized main cleanup into the compact branch. Resolve overlaps by composing the new
  cleanup with the compact branch's existing behavior in exactly these pre-run delta files:
  `docs/FABLE.md`, `src/fable/_runtime.py`, `src/fable/evaluation.py`,
  `src/fable/modeling.py`, `src/fable/temporal.py`, `tests/evaluation/test_evaluate.py`,
  `tests/temporal/test_history.py`, and `tests/test_modeling.py`.
- Preserve the compact branch's device-resident batching/data-loader semantics and its associated
  tests. Do not transplant main's CPU-loader implementation over them or widen the compact delta.
- Commit the reconciliation and record both parent refs. Do not edit this ledger in the worker.

### Focused checks

- Compare the reconciled branch against finalized `main` by both filename and hunk; only the
  approved compact data-loader/CUDA behavior may differ.
- Run root Ruff check/format, Pyright, Vulture with manual review, and the full Python suite.
- Run full app Vitest and TypeScript checks if the merge changes app-visible files.
- Run `git diff --check`, verify the merge ancestry, and audit exact branch/worktree/status state.

### Non-goals and gates

- No new CUDA behavior, scientific change, GPU execution, Slurm activity, remote checkout, research
  image build, app asset generation, or branch rewrite.
- No new branch or worktree. The existing compact branch is user-owned and must remain intact.

### Implementation-review record

- Implementer: `/root/compact_reconcile` using the `implement` skill.
- Reconciliation head: `60f73d4bad3781b59f7e8e0886f3197e076da30a`
  (`merge(cuda): align architectural cleanup`), with exact parents `bc9532cf` and `44b309ee`.
- Conflicts: `docs/FABLE.md` and `src/fable/temporal.py`. Resolution applied the shared cleanup and
  TF32 wording while preserving device-resident backing, integer-index loader/collation, shared
  device transfer, and their tests.
- Implementer checks: focused compact suite 15 passed; full Python 84 passed; Ruff check/format,
  Pyright, Vulture, full app Vitest 36, TypeScript, canonical mobile-export 11, diff, ancestry,
  branch, worktree, and status checks passed.
- Reviewer: `/root/compact_review` using the `code-review` skill, with independent parallel Standards
  and Spec axes over the fixed `bc9532cf...60f73d4` merge range and explicit hunk comparison to
  finalized main.
- Standards: zero actionable findings and zero smells. Conflict resolutions correctly compose the
  cleanup with compact batching; CPU worker/pinning/prefetch behavior did not return.
- Spec: zero actionable findings. Both exact parents are ancestors, every reviewed main cleanup is
  present, compact behavior was not widened, and the remaining main delta is exactly the approved
  eight files.
- Correction rounds: none.
- Final result: `GREEN LIGHT`. Reviewer made no mutations; only protected
  `?? docs/experiments/` remained.
- Orchestrator compact integration: full Python 84, app 36, mobile-export 11, Ruff check/format,
  Pyright, TypeScript, Vulture, documentation/conflict residue, diff, ancestry, filename/hunk,
  branch, worktree, and status audits all passed. Only dependency and expected local Lightning
  warnings were emitted.
- Intentionally unrun: GPU execution/training, Slurm/SSH, live RPC, remote checkout/image build,
  generated assets, native app build, simulator/device and visual acceptance.

## Final integration and cleanup

After Slices 1–3 are green, run steps 1–4 on `main`; after Slice 4 is green, repeat every applicable
check on the compact branch and finish steps 5–8:

1. Run root Ruff check/format, Pyright, Vulture with manual review, and the full Python suite.
2. Run the full app Vitest suite and TypeScript typecheck.
3. Run the canonical isolated mobile-export suite if Slice 3 changes the exporter.
4. Search tracked documentation for stale names or semantics; run `git diff --check`; verify exact
   branch, worktree, and status state; record every intentionally unrun external gate.
5. Record all immutable refs, workers, reviews, corrections, checks, and results in this ledger.
6. Delete this temporary ledger on `main` in a final orchestration commit, then propagate that
   ledger-only deletion to `codex/compact-cuda-execution` without changing product files.
7. Verify the compact branch contains finalized `main` and that their remaining product diff is
   exactly the approved compact data-loader/CUDA delta.
8. Return the sole checkout to `main`; verify no run-owned branch/worktree exists and only the
   pre-existing unrelated `docs/experiments/` status remains.

# FABLE architectural simplification ledger

Status: planning; implementation not started.

This ledger is the authoritative local specification for the approved architectural
simplification run. The orchestrating thread owns this file. Implementers and reviewers must not
edit it.

## Run policy

- Use the implementation-review loop: a fresh implementer for each slice, then a distinct fresh
  reviewer over fixed baseline and head commits. The reviewer evaluates Standards and Spec as
  separate axes through the repository `code-review` skill. A slice advances only after both axes
  have zero actionable findings.
- Return rejected findings to the same slice implementer. Send correction commits to the same
  reviewer. Review only the correction range against the outstanding findings.
- Preserve unrelated work. Do not push, open a pull request, mutate remote systems, submit Slurm
  jobs, or build research images. Alter the compact-CUDA branch only in Slice 3 and final ledger
  cleanup.
- Use a clean break. Add no compatibility alias, deprecated loader, legacy shim, registry,
  migration, or architectural transition test.
- Keep this ledger temporary. After every authorized slice is green and final integration checks
  pass, delete it in the run and verify that repository branch/worktree state matches the pre-run
  state except for the approved product commits.
- Work directly in the shared checkout. Create no branch or worktree, and allow only one writer at
  a time. Slices 1 and 2 commit on `main`; Slice 3 switches the same checkout to the pre-existing
  `codex/compact-cuda-execution` branch for final reconciliation.

## Pre-run state

- Recorded: 2026-07-31, Europe/Rome.
- Repository: `/Users/edo/dev/python/fable`.
- Branch: `main`.
- Immutable planning baseline: `92232703d2bf29cc26b6003c4a09e3146715d628`.
- Existing branches:
  - `main` at `92232703d2bf29cc26b6003c4a09e3146715d628`;
  - `codex/compact-cuda-execution` at `852d06a6f83646d170449f2554ce88bdeccdb5fc`.
- Existing worktrees: only `/Users/edo/dev/python/fable`, checked out on `main`.
- Compact-CUDA relation at planning baseline:
  - its merge base with `main` is exactly `92232703d2bf29cc26b6003c4a09e3146715d628`;
  - its intentional delta is restricted to `docs/FABLE.md`, `src/fable/_runtime.py`,
    `src/fable/evaluation.py`, `src/fable/modeling.py`, `src/fable/temporal.py`,
    `tests/evaluation/test_evaluate.py`, `tests/temporal/test_history.py`, and
    `tests/test_modeling.py`;
  - the planning-baseline delta SHA-256 is
    `257a49dfdf7082a24351c10e4d00b35046c307c11885490723050648e957e98a`;
  - those files express the accepted device-resident CUDA batching path in place of `main`'s
    data-loader path. The hash is evidence of the starting state, not a requirement to preserve
    stale code through the new cleanup.
- Pre-existing unrelated status: untracked `docs/experiments/`. It is user-owned and outside this
  run.
- Run-owned branches/worktrees: none.
- Planning-baseline verification:
  - `uv run vulture`: passed with no findings;
  - `uv run pytest -q`: 94 passed;
  - from `app/`, `npm test -- --run`: 35 passed;
  - from `app/`, `npm run typecheck`: passed.

### Transformer state-key planning record

Representative definitions use context `3`, feature count `2`, actions `2`, model width `8`, one
Transformer layer, feedforward width `16`, head width `4`, and zero dropout. The hybrid uses LSTM
hidden width `5` and one recurrent layer. `positions` is correctly absent because it is a
non-persistent buffer.

Common Transformer keys and shapes:

```text
projection.weight                                      (8, 2)
projection.bias                                        (8,)
encoder.layers.0.self_attn.in_proj_weight              (24, 8)
encoder.layers.0.self_attn.in_proj_bias                (24,)
encoder.layers.0.self_attn.out_proj.weight             (8, 8)
encoder.layers.0.self_attn.out_proj.bias               (8,)
encoder.layers.0.linear1.weight                        (16, 8)
encoder.layers.0.linear1.bias                          (16,)
encoder.layers.0.linear2.weight                        (8, 16)
encoder.layers.0.linear2.bias                          (8,)
encoder.layers.0.norm1.weight                          (8,)
encoder.layers.0.norm1.bias                            (8,)
encoder.layers.0.norm2.weight                          (8,)
encoder.layers.0.norm2.bias                            (8,)
```

Plain Transformer head keys and shapes:

```text
heads.action.0.weight                                  (4, 8)
heads.action.0.bias                                    (4,)
heads.action.3.weight                                  (2, 4)
heads.action.3.bias                                    (2,)
heads.regression.0.weight                              (4, 8)
heads.regression.0.bias                                (4,)
heads.regression.3.weight                              (1, 4)
heads.regression.3.bias                                (1,)
```

Transformer-LSTM tail and head keys and shapes:

```text
lstm.weight_ih_l0                                      (20, 8)
lstm.weight_hh_l0                                      (20, 5)
lstm.bias_ih_l0                                        (20,)
lstm.bias_hh_l0                                        (20,)
heads.action.0.weight                                  (4, 5)
heads.action.0.bias                                    (4,)
heads.action.3.weight                                  (2, 4)
heads.action.3.bias                                    (2,)
heads.regression.0.weight                              (4, 5)
heads.regression.0.bias                                (4,)
heads.regression.3.weight                              (1, 4)
heads.regression.3.bias                                (1,)
```

Before every slice, replace the proposed baseline with the actual immutable baseline, record exact
status, and verify that only known unrelated files are present.

## Governing sources

- `AGENTS.md`.
- `docs/CONTEXT.md`.
- `docs/FABLE.md`.
- `docs/adr/0006-direct-durable-object-authority.md`.
- `docs/adr/0007-native-external-execution-boundary.md`.
- `docs/agents/issue-tracker.md`.
- The approved decisions recorded below. This ledger is self-contained; the
  original finder audit is evidence, not authority.

## Decisions already aligned

### Approved changes

1. Remove the shallow in-memory `Corpus` wrapper as a clean break.
   - Replace the current `load_corpus(...) -> Corpus` interface with a direct block loader returning
     `BlockFrame`; use the clearest final name consistently, with `load_corpus_blocks` as the
     approved default.
   - Historical preparation accepts `BlockFrame` directly. Callers no longer reach through
     `.blocks`.
   - Keep `load_corpus_request` as the request owner used independently by held-out authoring and
     mobile export.
   - Keep canonical `corpus.json` plus `blocks.parquet` layout, request UUID verification, exact
     Parquet schema validation, `BlockFrame` isolation, range validation, and finalized-anchor
     producer bytes unchanged.
   - Remove the explicit `strict=True` argument from `load_corpus_request`; strictness remains owned
     by `StrictFrozenRecord` and must still reject coercive JSON.
   - Update every public prose/signature reference. Do not leave documentation describing the
     deleted `Corpus` value.

2. Deepen `_raw_feature_rows` around `BlockFrame`.
   - Accept `BlockFrame`, unwrap its Polars frame and chain ID internally, and let both public
     callers pass their existing `BlockFrame` directly.
   - Reuse `_feature_predecessor_blocks` so predecessor ownership has one implementation.
   - Preserve feature order, every feature formula, float64 assembly, contiguous output, conditional
     one-block interval predecessor, float32 transformation, and finite-output behavior.
   - Compact mean and standard-deviation conversion only where strict Python-float hydration remains
     identical.

3. Merge the Transformer implementation hierarchy.
   - Replace `_TransformerBackbone`, `_TransformerModel`, and `_TransformerLstmModel` with one
     `_TransformerModel` accepting `TransformerDefinition | TransformerLstmDefinition` and owning an
     optional LSTM tail.
   - Share one private LSTM constructor between the plain LSTM model and Transformer-LSTM tail.
   - Merge the two Transformer construction arms in `_FitModule`.
   - Preserve projection, non-persistent positions, encoder, optional recurrent tail, two heads,
     BF16 Transformer execution, explicit float32 recurrent execution, outputs, and checkpoint
     parameter names and shapes.
   - Flatten encoder initialization over `encoder.parameters()` only if the constructed encoder has
     no separate normalization module and initialization remains byte-for-byte equivalent in
     meaning.
   - Do not add a compatibility class or checkpoint migration path.

4. Remove the dead HPO Study cache.
   - Deduplicate the nine authored rows per cell into one ordered `cell -> study_id` mapping.
   - Load each cell's Study once, select its best result, and fully resolve all selections before
     publishing the manifest.
   - Preserve first-cell insertion order, printed row order and formatting, best-result tie behavior,
     manifest order, and fail-before-publication behavior.
   - Do not derive Transformer-LSTM capacity data from Transformer capacity data. The independent
     frozen scientific tables remain explicit.

5. Apply the approved Python test cleanup without weakening contracts.
   - Factor repeated valid Transformer and Tune payload construction in `tests/test_config.py`, but
     keep FABLE-owned invalid cases: Transformer dimensions, duplicate Methods, mixed model
     families, temporal-window ordering, and unsupported feature names.
   - Keep strict JSON round-trip coverage and consolidated UUIDv4 destination coverage.
   - Remove repeated UUIDv4 assertions only after representative unit and end-to-end coverage still
     proves request destinations and experiment identifiers.
   - Keep one representative manifest-only publication assertion for the shared bundle publisher;
     remove identical copies that cannot exercise a different owner.
   - In `test_k_study.py`, replace the vacuous typed-source filter with direct access and remove the
     length assertion already proven by the row count.
   - Keep `test_load_study_rejects_non_strict_json`.
   - Replace the four unrolled rolling-evaluation fixture publications with one explicit
     horizon/action loop.
   - Test remote Train and Evaluate dispatch in one straight-line test with branch-free assertions;
     retain small local helpers when they improve readability.
   - Remove `REMOTE.yaml`, cwd, sbatch stub, and fixture-inequality ceremony from duplicate allocation
     tests. The expected uniqueness failure must occur before any submission configuration access.
   - Do not add a global weakly typed Method factory. Keep the manifest-loader tests, independent
     held-out feature golden, named observation rows, and ragged-batch numerical aggregation test.

6. Apply all six approved app simplifications.
   - Move the shared mobile model manifest/selection fixture to `app/test/helpers.ts`. Use type-only
     imports so tests that do not mock ExecuTorch cannot load the native module. Use the inference
     fixture's feature and target constants because its assertions depend on them; model-runtime
     assertions must remain unchanged.
   - Consolidate only the six intentional style primitives: header row, card row, section gap,
     clipped card, last row, and teal/accent text. Use names describing shared visual intent. Do not
     merge the three pairs identified as merely coincidental.
   - Type the App test's captured mock props once at storage and delete the three cast-only accessor
     functions. Runtime behavior stays unchanged.
   - Make `chartScale` return one flat Gifted Charts prop bag with consumer-owned property names.
     Preserve negative and nonnegative chart geometry; perform a visual check in addition to tests
     when a runnable local screen is available.
   - Gate `RunDetails` at its caller and make its `run` prop non-null, matching the sibling overlay
     pattern.
   - Use the non-null disposal promise as the single disposed-state owner in the model runtime.
     Preserve immediate execute-after-dispose rejection, idempotent disposal, queued execution,
     model replacement, and native deletion order.

7. Remove `FeatureState.validate_widths` under the bounded supported-artifact model.
   - Normal production derives means and standard deviations from one raw feature matrix, so their
     widths cannot differ.
   - Supported artifacts come only from the canonical experiment pipeline or reviewed Codex
     changes. Hand-authored associations, alternate producers, and compatibility inputs do not
     exist in scope.
   - Delete the width validator, validator-only imports, and its dedicated hand-construction
     rejection test.
   - Keep nonempty, finite, positive state fields and `ArtifactAssociation`'s feature-count
     association check. This explicitly accepts that unsupported altered checkpoint bytes could
     broadcast a malformed standard-deviation vector.

8. Remove `_load_evaluation` request-identity and ordered-origin revalidation.
   - Trust the canonical atomic publisher to keep `evaluation.json` paired with its complete ordered
     observations. Fixed evaluation data is not manually copied, mixed, truncated, or edited.
   - Both reducers read canonical observations directly. Keep exact Parquet schema validation
     because conversion and metric columns depend on it.
   - Remove request/window-only imports and rejection tests. Keep exact `evaluation.json`
     publication as durable provenance.
   - Amend ADR 0006 and `docs/FABLE.md` in the same slice: a completed evaluation still owns its
     exact request and observations, while transient reduction validates the observation schema and
     trusts atomic publication for request pairing and ordered coverage.

### Protected non-goals

- Keep the named hour and day-of-week feature functions. Do not introduce a generic cycle helper.
- Keep workflow/candidate allocation uniqueness guards and their focused tests.
- Keep independent HPO capacity tables.
- Keep the explicit `reduce_baselines` accumulator loop.
- Keep `TuneRequest.method_at` and `CandidateProcessInput` upper-bound validation.
- Keep reuse of the already-loaded first candidate result during Study publication.
- Keep exact feature formulas, temporal geometry, target/loss/decode semantics, validation economics,
  rolling comparison semantics, checkpoint publication, candidate resume, and all atomic no-clobber
  behavior.
- Keep all other raw input, RPC, native tensor, schema, and durable-object checks.
- Do not touch `docs/experiments/`, mobile export, research runtime, Slurm configuration, or queued
  jobs. Slices 1 and 2 do not touch the compact-CUDA branch; Slice 3 changes it only through the
  bounded reconciliation recorded below.

## Slice 1 — Python owners, orchestration, documentation, and focused tests

Status: green.

Proposed baseline: planning baseline plus the finalized ledger commit. Record the immutable SHA
immediately before dispatch.

### Expected outcome

Python callers operate directly on the canonical `BlockFrame`; feature construction and Transformer
modeling have one clear owner each; HPO selection carries no dead cache; documentation and focused
tests describe the final clean-break interfaces without weakening scientific, durable-object, or
submission contracts.

### Scope

1. Implement approved decisions 1–5, 7, and 8 above, including all related imports, type
   annotations, tests, and `docs/FABLE.md` references.
2. Resolve every affected caller of the deleted `Corpus` interface across `src/`, `tests/`, and
   experiment/evaluation code. Keep independent `load_corpus_request` callers unchanged except for
   import cleanup.
3. Preserve Transformer and Transformer-LSTM checkpoint state names and shapes. Baseline reference
   shapes were captured during planning for representative definitions; the implementation must
   demonstrate identical projection, encoder, optional `lstm`, and `heads` key paths.
4. Amend ADR 0006 exactly as required by approved decision 8. Do not weaken any other durable-object
   authority or publication rule.
5. Format only touched Python files. Make no unrelated mechanical rewrite.

### Focused checks

- Corpus loader and blocks: `uv run pytest -q tests/corpus/test_corpus.py tests/corpus/test_blocks.py`.
- Temporal features/history: `uv run pytest -q tests/temporal/test_features.py tests/temporal/test_history.py`.
- Modeling: `uv run pytest -q tests/test_modeling.py`.
- HPO and changed orchestration tests: targeted files under `tests/experiments/`,
  `tests/test_execution.py`, `tests/cli/test_study.py`, and `tests/cli/test_remote_workflow.py`.
- Capture representative Transformer and Transformer-LSTM `state_dict` names/shapes and compare
  them with the planning record.
- Run the existing Transformer-LSTM float32 recurrence/export test after the merge.
- `uv run ruff check` over touched Python paths and `uv run pyright` during implementation.

### Slice integration checks

- `uv run ruff check .`.
- `uv run ruff format --check .`.
- `uv run pyright`.
- `uv run vulture`, followed by manual validation of any finding.
- `uv run pytest -q`.
- Documentation search proving no stale in-memory `Corpus` interface or deleted private Transformer
  class remains.
- `git diff --check` and exact status/mutation audit.

### Non-goals and gates

- No app changes; those belong to Slice 2.
- No GPU training, Slurm submission, research image build, remote checkout change, or mobile-export
  execution.
- No real stored Transformer checkpoint is available as a required external gate. State-key/shape
  equivalence plus existing construction/export checks are the local acceptance evidence.

### Implementation-review record

- Baseline: `59766c072b6dd4d213e7ce243091850e8d15053d`.
- Implementer: `/root/slice1_python`.
- Implementation head: `acff461676270da1c083a87996791a3414b700f3`.
- Focused/integration checks: corpus 11 passed; temporal 10 passed; modeling/export 7 passed;
  evaluation 11 passed; orchestration/config/execution 24 passed; full suite 89 passed; Ruff check
  and format passed; Pyright passed; Vulture passed with no findings; `git diff --check` passed;
  representative Transformer and Transformer-LSTM state keys/shapes matched the planning record.
  The orchestrator reran the full suite: 89 passed with 15 expected local Torch/Lightning warnings.
- Reviewer: `/root/slice1_review`, with parallel `/root/slice1_review/standards_axis` and
  `/root/slice1_review/spec_axis`.
- Standards findings: 0.
- Spec findings: 0.
- Correction rounds: 0.
- Explicitly not run: GPU training, Slurm submission, research image build, remote changes, mobile
  export, and app checks.
- Final result: `GREEN LIGHT`; read-only review pinned the exact baseline and implementation head.

## Slice 2 — App fixture, presentation primitives, and runtime state

Status: green.

Proposed baseline: Slice 1 final green head plus the orchestrator's ledger update. Record the
immutable SHA immediately before dispatch.

### Expected outcome

The app expresses shared test data, visual primitives, chart configuration, overlay presence, and
model disposal through one owner each while preserving the rendered behavior and serialized native
runtime semantics.

### Scope

1. Implement approved decision 6 exactly.
2. Keep ExecuTorch imports erased from general test helpers at runtime.
3. Preserve the inference test's manifest values and every existing model-runtime assertion.
4. Consolidate only deliberately shared styles. Choose names that state shared visual meaning and
   update every caller atomically.
5. Flatten chart props without altering positive/negative scaling or data construction.
6. Remove the nullable `RunDetails` branch only after caller gating makes non-nullness structural.
7. Remove `disposed` only after `disposal !== null` rejects new execution synchronously and repeated
   disposal still returns the same promise.

### Focused checks

- `npm test -- --run test/model.test.ts test/inference.test.ts`.
- `npm test -- --run test/App.test.tsx`.
- Targeted Analytics/app test files covering touched screens and styles.
- `npm run typecheck` during implementation.
- Visual inspection of all three Analytics charts when a runnable local screen is available. If no
  runnable screen is available, record the check as explicitly not run; do not claim visual parity.

### Slice integration checks

- From `app/`, `npm test -- --run`.
- From `app/`, `npm run typecheck`.
- Root `git diff --check` and exact status/mutation audit.

### Non-goals and gates

- No Python, scientific, RPC, inference-engine lifecycle, history persistence, native asset,
  mobile-export, Expo configuration, or dependency-version changes.
- No iOS/Android native build or real-device acceptance unless separately authorized.

### Implementation-review record

- Baseline: `922296bc41c51d5aa5fe75661692bd9d96573597`.
- Implementer: `/root/slice2_app`.
- Implementation head: `be78019f0e381f1553238745adb0eceab9c20126`.
- Focused/integration checks: model/inference 14 passed; App/Analytics 7 passed; full app suite
  35 passed; TypeScript typecheck passed; `git diff --check` passed. The orchestrator reran the full
  app suite and typecheck with the same results.
- Reviewer: `/root/slice2_review`, with parallel `/root/slice2_review/standards_axis` and
  `/root/slice2_review/spec_axis`.
- Standards findings: 0.
- Spec findings: 0.
- Correction rounds: 0.
- Explicitly not run: visual inspection of the three Analytics charts. The available simulator has
  Expo Go only; FABLE requires a development build containing its custom ExecuTorch native module.
  A native build was outside scope. The reviewer found no code-evidenced parity risk.
- Final result: `GREEN LIGHT`; read-only review pinned the exact baseline and implementation head.

## Slice 3 — Compact-CUDA reconciliation

Status: ready; Slices 1 and 2 are green.

Proposed refs: the then-current `main` head and the pre-existing compact-CUDA head. Record both
immutable SHAs immediately before dispatch. Work in the shared checkout with one writer, switch to
`codex/compact-cuda-execution`, and merge the reviewed `main` head. Create no branch or worktree.

### Expected outcome

The compact-CUDA branch contains every applicable reviewed cleanup from `main`. Its only remaining
semantic difference is the existing device-resident CUDA batching implementation in place of
`main`'s data-loader implementation; no stale general architecture, tests, documentation, or app
code remains.

### Scope

1. Merge the exact reviewed `main` head into `codex/compact-cuda-execution` with a merge commit.
2. Apply every cleanup unchanged where the branch does not intentionally diverge.
3. Resolve overlaps in the eight accepted divergence files around the branch's canonical CUDA
   owners. Preserve integer-index `HistoricalDataset.__getitem__()` and device-side `_batch()`
   tensor mapping rather than reintroducing `main`'s data-loader collation.
4. Preserve the final public and scientific interfaces established by Slices 1 and 2: direct
   `BlockFrame` corpus loading, feature ownership, merged Transformer structure and checkpoint
   keys, evaluation schema-only transient loading, revised ADR authority, HPO selection, and app
   simplifications.
5. Update compact-CUDA documentation and tests to describe and test only current branch behavior.
   Do not preserve stale main or CUDA implementation names merely to reduce merge conflicts.
6. Keep the post-merge branch delta against `main` restricted to the same eight accepted files
   recorded in the planning baseline. Inspect actual hunks, not filenames alone. A changed patch
   hash is expected when shared cleanups alter those files.

### Focused checks

- Run the Slice 1 focused Python tests that touch any of the eight divergence files.
- Capture representative Transformer and Transformer-LSTM state keys/shapes and compare them with
  the planning record.
- Run the compact-CUDA historical batching, evaluation, modeling, and runtime tests covering
  integer indices, device-side collation, float32 recurrence, and evaluation reduction.
- Run app tests touching Slice 2 changes from `app/`.
- Inspect `git diff main...HEAD` and prove every hunk is CUDA/data-loader-specific rather than a
  missed general cleanup.

### Slice integration checks

- `uv run ruff check .`.
- `uv run ruff format --check .`.
- `uv run pyright`.
- `uv run vulture`, followed by manual validation of any finding.
- `uv run pytest -q`.
- From `app/`, `npm test -- --run` and `npm run typecheck`.
- `git diff --check`, exact eight-file delta audit against `main`, and exact status/mutation audit.

### Non-goals and gates

- Do not change the accepted CUDA batching design, introduce a second runtime path, or copy
  `main`'s data-loader mechanics into the compact branch.
- Do not add compatibility code for pre-cleanup names or checkpoint shapes.
- No GPU training, Slurm submission, research image build, remote checkout change, native build,
  or real-device acceptance.
- GPU smoke testing remains an external gate and is not required for this architectural merge.

### Implementation-review record

- Main ref: pending.
- Compact baseline: pending.
- Implementer: pending.
- Implementation head: pending.
- Focused/integration checks: pending.
- Reviewer: pending.
- Standards findings: pending.
- Spec findings: pending.
- Correction rounds: pending.
- Final result: pending.

## Final integration and cleanup

After all three slices are green:

1. On both `main` and `codex/compact-cuda-execution`, run root `uv run ruff check .`,
   `uv run ruff format --check .`, `uv run pyright`, `uv run vulture`, and `uv run pytest -q`.
2. On both branches, run `npm test -- --run` and `npm run typecheck` from `app/`.
3. Verify documentation, exact checkout status, branches, worktrees, and untouched
   `docs/experiments/`. Prove the compact branch contains `main` and differs only in the exact eight
   accepted CUDA/data-loader files, with every remaining hunk inspected as intentional.
4. Record all slice refs, workers, review results, correction rounds, checks, explicitly unrun gates,
   and the final compact delta audit in this ledger; commit that record on the compact branch. Then
   delete the ledger there and commit the deletion.
5. Switch to `main`, delete this temporary ledger, and commit the deletion. Switch to the compact
   branch, merge that exact `main` cleanup commit, and verify the ledger is absent from both branch
   tips.
6. Leave the shared checkout on `main`. Verify no run-owned branch/worktree exists and that the
   branch/worktree state matches the pre-run state except for the authorized commits on `main` and
   `codex/compact-cuda-execution`.
7. Report completion only if every slice has a zero-finding green review, both branches pass final
   integration, compact-CUDA contains `main`, its delta is bounded exactly as required, and the
   ledger has been removed from both branch tips.

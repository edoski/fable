# FABLE residual leanness ledger

Status: execution in progress. Slices 1 and 2 are complete and green.

## Authority

- Proposal: `wf_d2060434-241`, supplied as
  `/Users/edo/.codex/attachments/be7b40c3-4158-43f1-8fdf-835ea0e74c50/pasted-text.txt`.
- Standards: `AGENTS.md`.
- Domain and durable contracts: `docs/CONTEXT.md`, `docs/FABLE.md`,
  `docs/adr/0006-direct-durable-object-authority.md`, and
  `docs/adr/0007-native-external-execution-boundary.md`.
- Workflow: `implrevloop`; each slice gets a fresh implementer using `implement`, then a distinct
  read-only reviewer using `code-review`. A slice advances only after both Standards and Spec return
  zero actionable findings.

## Initial state

- Baseline: `bb47f518b912ce91e4f7d1c7a543ad66006e11d5`.
- Checkout: `main`, ahead of `origin/main` by 32 commits.
- Existing worktree: `/Users/edo/dev/python/fable` only.
- Existing branches: `main` and `codex/compact-cuda-execution`.
- Protected unrelated state: untracked `docs/experiments/`.
- No run-owned branch or worktree exists. No external system may be mutated.

Before each slice, record its immutable baseline, branch, worktrees, status, implementer, and
reviewer. Preserve `docs/experiments/` byte-for-byte.

## Run isolation

- Approved-ledger commit: `52a816c5c7c1c573bab6752627d405fd548d246d`.
- Run-owned branch: `codex/residual-leanness`.
- Run-owned worktree: `/Users/edo/dev/python/fable-residual-leanness`.
- The run branch and worktree were created from the approved-ledger commit.
- The original `/Users/edo/dev/python/fable` checkout remains on `main`; its untracked
  `docs/experiments/` is outside the run worktree and remains protected.
- No other branch or worktree is run-owned.

## Decisions

- Keep the run compressed to four ordered slices.
- Use one isolated run branch and worktree because no direct-main checkout override was requested.
  Integrate the reviewed linear history into `main`, then remove only the run-owned branch and
  worktree.
- Slice 1 has an explicit user waiver from independent review because it is pure formatting. It
  still requires a fresh implementer, AST equivalence, the full listed checks, orchestrator diff
  verification, and a clean immutable commit. Slices 2–4 retain the complete independent
  implementation-review-correction loop.
- Prefer fewer lines only when the result is also cleaner, more direct, idiomatic, and less
  contrived. Do not trade visible lines for hidden machinery.
- Accept modest CPU memory or runtime cost only when the resulting code is materially simpler.
- Let `_build_dataset` always derive its own outcomes. The extra chunked training pass is bounded
  and negligible relative to fitting; remove the one-caller `outcomes` exception.
- Use one generic `close_bundle` for the four current close workflows. This consolidates existing
  behavior in a fixed domain; it is not an extensibility point. Keep the known record columns and
  verifying loaders explicit at each call site. Add no registry, fallback, shim, or compatibility
  path. The helper should be the direct shared loop plus one-line fixed call sites.
- Do not use the proposed `C x N` F1 broadcast. Held-out evaluation reaches `K=200`, and reduction
  runs in CPU NumPy rather than on the GPU. Replace the loop with an equivalent `np.bincount`
  reduction whose memory is `O(N)` and whose class set remains the observed union. Use the direct
  count-vector formula; add no chunking or alternate reducer.
- Treat formatter compaction as a formatting policy, not a semantic cleanup. Keep it isolated from
  product edits.
- Keep these two rejected proposals as explicit non-goals:
  1. dataclass conversion of `HistoricalDataset`: the explicit constructor remains the owner of
     NumPy-to-Torch conversion and preserves ordinary identity semantics; generated tensor equality
     fails, generated representation exposes large tensors, and disabling those dataclass behaviors
     would make the shorter declaration more contrived;
  2. inlining `resolveOutcomes` into the block watcher.

## Accepted findings

Thirty-two findings are planned:

- Core numerics: formatter policy; uniform `_build_dataset` outcome derivation; inline
  `_natural_log`; direct `argmin` plus `min`.
- Training: shared epoch logger; direct DataLoader profile; omit pinned Lightning defaults; inline
  the one-use Study loader and Method-index alias; inline JSON association serialization.
- Evaluation: `O(N)` bincount classification metrics; formatter-owned call compaction; inline
  `_reduce`; direct byte loading with model-owned strictness.
- Orchestration: direct mobile-export entry point; shared experiment CLI runner; generic
  `close_bundle`; direct allocation validation/config loading; one `jobs.tsv` existence decision;
  inferred workflow-request list type.
- App: declarative `waitBuckets`; direct native-module cache return; SummaryCard-owned null display;
  one bounded chart scaffold.
- Tests: one modeling Method fixture; remove duplicate two-workflow renderer coverage; parameterize
  fee-history rejections; trim the model-runtime manifest fixture; remove the Pydantic round-trip
  test; simplify inference helpers; remove duplicate rolling-schema coverage; remove duplicate
  allocation-size cases; fold constant-feature rejection into the existing contract table.

## Slice 1: formatter policy

Status: complete under the user-approved review waiver.

### Execution record

- Baseline: `f57f165626dd87ed2258253da57a05e55cbecc25`.
- Implementer: `/root/slice1_formatter`.
- Implementation head: `9d9c5e3df9da5d49677dd33d905c288033b7c918`.
- Commit: `style: compact Python formatting`.
- Reviewer: waived explicitly by the user because this slice is pure formatting.
- Scope: `pyproject.toml` plus 42 governed Python files; 346 additions, 1,369 deletions,
  net -1,023.
- Implementer checks: 42 changed Python ASTs equivalent to baseline; Ruff format and lint passed;
  Pyright reported zero errors; Vulture passed; 123 tests passed.
- Orchestrator verification: exact baseline parent and head confirmed; worktree clean; diff scope and
  Ruff settings confirmed; 42 ASTs independently compared; Ruff format and lint, Pyright, and
  Vulture passed.
- Review result: waiver gate satisfied; no correction round.

### Scope

Add Ruff's skip-magic-trailing-comma and matching isort policy, then format all governed Python
sources, tests, experiments, and tools.

### Approved changes

1. **Repository-wide compact Ruff policy.** Add
   `format.skip-magic-trailing-comma = true` and the matching isort
   `split-on-trailing-comma = false` setting, then run Ruff formatting once across `src/`, `tests/`,
   `experiments/`, and `tools/mobile-export/`. The measured current effect is a net 935-line
   reduction. This is a style decision: only layout may change.
2. **Formatter-owned evaluation call compaction.** Let the new policy collapse the two long
   `_require_finite` and `_classification_metrics` calls in `evaluation.py`. Do not hand-format
   those calls or mix their compaction with the later evaluation refactor.

### Non-goals and protected behavior

- No hand-authored semantic edits.
- No changes to schemas, names, APIs, tests, or runtime behavior.
- Do not touch the app or protected unrelated state.

### Expected outcome

Python uses one self-enforcing compact layout. The same ASTs and behavior remain, with broad
vertical formatting removed in one reviewable mechanical commit.

### Checks

- `uv run ruff format --check`
- `uv run ruff check`
- `uv run pyright`
- `uv run pytest`
- `uv run vulture`
- AST-equivalence check across every reformatted Python file

Dependencies: none. External gates: none.

## Slice 2: Python scientific and runtime cleanup

Status: complete. Independent review returned zero Standards findings and zero Spec findings.

### Execution record

- Baseline: `e84797cca171826d85606c65fa3b045b8e42b793`.
- Implementer: `/root/slice2_python`.
- Implementation head: `fbe969bc614cc6a580f91e379034d900a8a36b59`.
- Commit: `refactor(science): simplify Python runtime`.
- Reviewer: `/root/slice2_reviewer`, with independent Standards and Spec axes.
- Scope: 11 Python files; 86 additions and 186 deletions, net -100.
- All 15 approved changes were implemented. `HistoricalDataset` remains explicit, and the
  classification reducer uses `O(N)` observation arrays plus count vectors rather than a
  `K x N` allocation.
- Implementer checks: 47 focused tests and 121 full-suite tests passed; Ruff format and lint,
  Pyright, and Vulture passed; 5,600 randomized old/new F1 comparisons were bit-identical through
  `K=200`.
- Reviewer checks: 52 focused tests and 121 full-suite tests passed; Ruff format and lint,
  Pyright, Vulture, and diff checks passed; an additional 1,120 randomized F1 comparisons were
  bit-identical through `K=200`. Lightning checkpoint defaults resolved to
  `(every_n_epochs=1, every_n_train_steps=0)`, and the one-row evaluation schema and column order
  matched the baseline.
- Review result: `GREEN LIGHT` with zero Standards findings and zero Spec findings; no correction
  round.

### Scope

Implement the accepted core-numerics, training, and evaluation findings, including the revised
`np.bincount` F1 reduction. Apply the associated modeling/evaluation test cleanup.

### Approved changes

1. **Uniform dataset outcome derivation.** Remove `_build_dataset(outcomes=...)` and its conditional
   path. `prepare_fit_history` computes training minima for `TargetState`; `_build_dataset` then
   derives the labels and minima it owns like every other caller. This accepts one repeated chunked
   calculation to remove the special-case parameter and cross-step coupling.
2. **Direct target logarithms.** Delete `_natural_log` and call `np.log(raw_minima)` directly in
   `fit_target_state` and `standardize_target`. NumPy already returns `float64` for the canonical
   `int64` input, so the helper and explicit no-copy cast add no behavior.
3. **Direct chunk minima.** Store `outcomes.argmin(axis=1)` as labels and
   `outcomes.min(axis=1)` as minima. Remove the temporary label array, `np.arange`, and fancy-index
   gather. Preserve first-minimum action selection and exact minimum values.
4. **One epoch-metric logger.** Replace `_log_epoch_loss(role, values)` with one
   `_log_epoch(name, values)` helper used by training loss, validation loss, and validation
   base-fee optimality gap. Preserve float64 mean reduction, detachment, metric names, epoch-only
   logging, disabled logger forwarding, and batch-size weighting.
5. **Direct DataLoader runtime profile.** Keep `NUM_WORKERS`, `FIT_BATCH_SIZE`, and
   `EVALUATION_BATCH_SIZE`; inline the one-use pin-memory, prefetch, and persistent-worker facts.
   Remove the `workers` alias and explicit `drop_last=False` default. Tests must retain the ability
   to set `NUM_WORKERS=0`.
6. **Use the pinned Lightning checkpoint default.** Remove both `every_n_epochs=1` arguments.
   Lightning 2.6.5 produces the same `(every_n_epochs=1, every_n_train_steps=0)` triggers when the
   arguments are absent. Keep every non-default best/last checkpoint option and resume behavior.
7. **Inline one-use Study ceremony.** Load the canonical Study directly inside `load_study`, delete
   `_load_study_path`, and place `Annotated[int, Field(ge=0)]` directly on
   `_CandidateResult.method_index`. Keep candidate-result loading separate because it has multiple
   callers and owns a distinct record type.
8. **Direct association serialization.** Delete `_json_association`; pass
   `association.model_dump(mode="json")` directly into `_FitModule` and its two test constructions.
   Keep `_hydrate_association`, including its JSON round-trip and strict union validation.
9. **Linear-memory classification reduction.** Replace the per-class F1 loop with one direct
   `np.bincount` count-vector formula over canonical nonnegative integer actions. Compute truth,
   prediction, and true-positive counts; average only classes present in their union. Preserve exact
   accuracy and macro-F1 results without the rejected `K x N` broadcast allocation. The validated
   prototype matched the current implementation on randomized inputs through `K=200` and reduced a
   one-million-row K=200 probe from about 121 ms to about 4 ms.
10. **Deepen the public evaluation reducer.** Move `_reduce` into its sole caller,
    `reduce_evaluation`, and construct the one-row result with `pl.DataFrame([metrics])`. Preserve
    metric names, order, `Float64` schema, finite validation, and the separation between canonical
    observation loading and transient metric calculation.
11. **Direct canonical evaluation JSON loading.** Read `evaluation.json` as bytes and rely on
    `EvaluateRequest`'s inherited strict model configuration instead of repeating `strict=True`.
    Keep evaluation-ID identity, observation schema, null, and exact-origin validation unchanged.
12. **One modeling Method fixture.** Replace `_definition(...).method` and the one-file
    `tests.helpers.modeling_method()` with one module-level `_METHOD` in `test_modeling.py`. All
    affected tests exercise the same LSTM shape; retain `TrainingDefinition` where checkpoint
    association assertions genuinely use it.
13. **Remove the Pydantic self-round-trip test.** Delete
    `test_artifact_association_round_trips_strict_json`. Artifact loading already drives strict
    hydration through the actual checkpoint boundary; keep the FABLE-specific association-width
    rejection and end-to-end artifact assertions.
14. **Remove duplicate rolling schema coverage.** Drop only the canonical-schema case from the
    rolling rejection parametrization because both public reduction paths call the same
    `_read_observations` validator. Keep rolling-specific consecutive-origin, missing-origin, and
    action-range cases.
15. **Unify feature-contract rejection coverage.** Move the standalone constant-feature rejection
    into the adjacent `test_feature_contract_rejections` parameter table with the explicit
    `constant-feature` ID and positive-standard-deviation message. Reuse the shared assertion body
    without changing the block fixture or production validation.

### Non-goals and protected behavior

- Keep `HistoricalDataset` as the direct tensor-owning class.
- Preserve exact target values, tie behavior, checkpoint resume/best semantics, metric names and
  weighting, evaluation schemas and column order, finite guards, request identity, and durable
  object validation.
- The F1 result must remain bit-identical on representative and randomized valid action arrays.
- Accept the bounded repeated training-outcome calculation, the extra pass for direct chunk minima,
  and bounded `O(N)` temporary classification arrays. No `O(K x N)` allocation.

### Expected outcome

Scientific and training code expresses each calculation and framework interaction once, with
fewer wrappers and defaults, while every numerical and durable contract remains unchanged.

### Checks

- Focused temporal, modeling, Study, evaluation, and affected experiment tests
- Randomized old/new classification equivalence through `K=200`
- `uv run ruff format --check`
- `uv run ruff check`
- `uv run pyright`
- `uv run pytest`
- `uv run vulture`

Dependencies: Slice 1 green. External gates: none.

## Slice 3: orchestration cleanup

### Scope

Implement all six orchestration findings and their associated test cleanup. Consolidate the four
current close workflows through one direct `close_bundle`.

### Approved changes

1. **Direct mobile-export entry point.** Move the `STORAGE_ROOT` Typer option annotation onto
   `export_bundle`, delete the forwarding `main`, and run `export_bundle` directly. Preserve the two
   positional paths, environment option, programmatic three-argument calls, and atomic hidden
   sibling export.
2. **One experiment CLI runner.** Add `run(*commands)` to `experiments/bundle.py`; use it from
   `c_study.py`, `feature_ablation.py`, `held_out.py`, `hpo.py`, and `k_study.py`. Delete their
   repeated Typer construction and now-unused imports. Keep `experiments/launch.py`'s module-level
   app because tests dispatch through it.
3. **One fixed-domain bundle closer.** Replace the Study-only closer and duplicated artifact and
   evaluation close loops with one `close_bundle(storage_root, kind, experiment_id, column,
   verify)`. Its direct loop reads cells, parses the known UUID column, invokes the supplied
   canonical loader/reducer, publishes the flat manifest, and prints the experiment ID. The four
   call sites must explicitly pair `study_id`/`load_study`, `artifact_id`/`load_artifact`, or
   `evaluation_id`/`reduce_evaluation`; add no registry or dynamic record discovery.
4. **Direct allocation setup.** Remove the redundant `tuple(inputs)`, inline the sole process-count
   guard and cwd-local `REMOTE.yaml` hydration into `_submit_allocation`, and delete the two one-use
   helpers. Keep count validation before file loading, `_invoke_sbatch` as the test seam, and exact
   generated script behavior.
5. **One launch-journal existence decision.** Snapshot whether `jobs.tsv` exists before loading
   submitted rows, then use that same fact for append/exclusive-create mode and header emission.
   Preserve sequential restart behavior and atomic row flush/fsync. This does not add or imply
   concurrent-launch support.
6. **Infer the workflow request list type.** Remove the redundant
   `list[WorkflowRequest]` annotation and its single-use import in `cli.py`. Keep the two-phase
   validate-all-then-submit structure so invalid later files cannot follow earlier submissions.
7. **Remove duplicate two-workflow renderer coverage.** Delete the N=2 workflow allocation test.
   Retain the exact N=1 workflow golden script, N=3 candidate scaling/order test, and both public
   duplicate-identity guards; together they cover the shared renderer without a Cartesian product.
8. **Trim repeated allocation-size cases.** Remove the `(5, 3)` and `(4, 3)` parametrizations.
   Surviving cases still cover the special singleton-avoidance split, ordinary remainder, exact
   capacity, single item, and capacity-two guard.

### Non-goals and protected behavior

- Keep exact CLI commands, argument order, environment variables, stdout, cell order, record IDs,
  verification calls, manifest bytes, and publication order.
- Keep two-phase workflow-request validation before submission.
- Keep launch restart behavior, append/flush/fsync ordering, allocation grouping, payload order,
  one-GPU steps, and exact log paths.
- Add no registry, new record kind, migration path, fallback, or concurrent-launch protocol.
- Do not invoke SSH, Slurm, or any external RPC.

### Expected outcome

Experiment and submission scripts share their genuinely identical lifecycle mechanics without
duplicated launch, close, or CLI ceremony. Their fixed scientific and scheduler behavior remains
observable at the same surfaces.

### Checks

- Focused execution, CLI, experiment, launch, and mobile-export tests
- CLI help/command-surface comparison for all affected scripts
- Manifest and stdout equivalence for all close workflows
- `uv run ruff format --check`
- `uv run ruff check`
- `uv run pyright`
- `uv run pytest`
- `uv run vulture`

Dependencies: Slice 2 green. External gates: none; external execution is forbidden.

## Slice 4: app and app-test cleanup

### Scope

Replace `waitBuckets` accumulator machinery with the declarative horizon-bucket reduction,
simplify the native-module cache return, centralize nullable summary display and the repeated chart
scaffold, and apply the accepted app-test fixture and parameterization cleanup.

### Approved changes

1. **Declarative wait buckets.** Replace the mutable bucket accumulator and `nextMean` helper with
   one `Array.from` over the fixed horizon. For each offset, select matching runs, collect valid
   outcomes, and use the existing `mean()` for savings and fee means. Preserve run order in every
   mean, count unresolved runs in `runCount`, exclude invalid outcomes from means, retain labels and
   Gwei conversion, and return `[]` when there are no runs.
2. **Return the native module directly.** Remove `LoadedModel` and the one-use `requireActive`.
   Keep the keyed `{key, module}` cache internally, but let `ensureLoaded` return `NativeModule`
   directly and call `forward` on it. Preserve synchronous disposed rejection, serialized
   replacement/disposal, deletion before replacement, load-failure cleanup, and output copying.
3. **Let SummaryCard own its null display.** Change `SummaryCard` from a preformatted string to a
   nullable number plus caller-supplied formatter. Render the em dash once inside the card and
   collapse the three repeated null ternaries at the call sites. Keep each metric's formatting
   function at the caller and preserve every rendered string.
4. **Use one bounded chart scaffold.** Fold `ChartFrame` and `EmptyGraph` into `ChartCard`. Let the
   card accept the title, optional legend, x-axis title, and exact empty state
   `"runs" | "outcomes" | null`; keep each named chart's data, scale, colors, and chart-specific
   properties in that chart. Preserve the rendered header, empty messages, graph frame, axis title,
   and view hierarchy. Add no generic chart configuration object or broad rest-prop abstraction.
5. **Parameterize fee-history rejection tests.** Merge the three repeated RPC fee-history rejection
   bodies into `it.each`, retaining each malformed history, exact error, and the P90-only case that
   proves priority-fee routing.
6. **Trim the model-runtime fixture to exercised data.** Replace the unused three-chain manifest and
   resource tables in `model.test.ts` with one Ethereum manifest and K-based source values. Preserve
   K=2/K=3 cache-key changes and exact module-load assertions; catalog-wide chain coverage is not
   the subject of this runtime test.
7. **Simplify inference test helpers.** Remove never-overridden manifest parameters from
   `selection` and `catalog`, and return `InferenceEngine` directly from `createTestEngine` instead
   of a one-field wrapper. Update the six call sites without changing any inference behavior or
   failure assertion.

### Non-goals and protected behavior

- Do not change `resolveOutcomes`, RPC validation, feature parity, overlays, selection currency,
  serialization, or persistence.
- Preserve wait-bucket labels, run counts, valid-outcome filtering, running-mean numerical order,
  null behavior, module replacement/disposal order, failure cleanup, tensor validation, and output
  copies.
- Accept `O(horizon x runs)` app aggregation with `horizon <= 5`.
- No visual redesign and no native asset regeneration.

### Expected outcome

Analytics and model-runtime code have less mutable bookkeeping and fewer one-use containers, while
the rendered values, persisted history semantics, native resource lifecycle, and tested error
behavior remain unchanged.

### Checks

- Focused analytics, model, inference, and RPC tests
- Randomized old/new wait-bucket JSON equivalence
- `npm run typecheck`
- `npm test -- --run`

Dependencies: Slice 3 green. External gates: no simulator or device check is required because the
accepted scope changes no visual structure.

## Execution gate

The user authorized execution on 2026-07-31. Before Slice 1, create and record the isolated run
branch/worktree from the approved ledger commit, then record the unchanged baseline/status. Do not
start a later slice until the preceding slice satisfies its gate.

# FABLE simplification implementation plan

**Status:** approved and implementation-ready
**Program baseline:** `0a4ad026b862da93b91877e69c509f30764f7293`
**Execution:** nine sequential, independently reviewed slices
**Deferred final-model work:** [A01 — Activate selected artifacts in the final Mac/mobile demo](https://github.com/edoski/fable/issues/139)

## Contract

- Keep one lean, direct, typed path that a single thesis developer can follow.
- Add no compatibility shim, migration, fallback, adapter, registry, speculative abstraction, or transition test.
- Trust typed values produced inside `src/` and the fixed app workflow.
- Retain checks only at raw-input, scientific/numerical, cross-job, atomic-publication, and native-runtime boundaries, plus demonstrated failures.
- Keep one canonical owner for each fact.
- Preserve unrelated worktree and index state.
- Preserve the untracked `docs/research/gpu-execution-optimizations.md` exactly where it is. Never stage, move, delete, or edit it.
- Run no experiment, RPC acquisition, Slurm job, generated-manifest workflow, deployment, or final model export in this program.
- Apply schema changes as clean breaks. Clear development state when required; add no migration.

## Implementer–reviewer protocol

Slices execute in order. A later slice starts only after the current slice is committed and GREEN.
Every implementer and reviewer task works directly in the saved project’s `main` checkout.
Create no worktree or slice branch.

For each slice:

1. Create one new xhigh implementer task.
2. The implementer records HEAD, `git status --short`, the index state, and the slice-owned paths before editing.
3. The implementer changes only the current slice, runs its focused checks, updates this plan’s slice checkboxes, and commits with the repository’s conventional prefix.
4. The implementer reports the baseline, commit, exact changed files, checks, final worktree/index state, and residual risks to this task.
5. Create one new xhigh, sole-authoritative reviewer task after the implementation commit exists.
6. The reviewer is read-only. It pins the exact baseline-to-candidate range, reviews every changed line and relevant owner/caller, runs the slice’s review checks, and reports:
   - actionable findings;
   - Standards verdict;
   - Spec verdict;
   - scope and mutation audit;
   - exact verification;
   - `GREEN LIGHT` or `RED LIGHT`.
7. Ignore any implementer-internal review. Only the parent-created reviewer is authoritative.
8. On RED, send the findings to the same implementer task. That implementer commits focused corrections. Send the new range to the same reviewer task. Repeat until GREEN.
9. Never let a reviewer fix, stage, commit, or broaden the slice.

Each reviewer verifies that the pre-existing GPU research note remains untracked and untouched. The last reviewer runs the cumulative matrix in addition to its slice-local checks.

## Slice 1 — Durable create-only publication

**Goal:** give Artifact, Study, and exporter outputs one no-clobber publication model.

### Implementation

- [x] Confirm hard-link support once on the real Slurm `STORAGE_ROOT`. On 2026-07-27, a bounded manual probe created two names with one inode and link count two, then removed both names and its temporary directory. This is setup evidence, not an automated test or recurring probe.
- [x] If hard links are unsupported, retain the current artifact-directory layout and stop the flat-artifact part of this slice until another atomic create-only design is approved. Not applicable because support is confirmed above.
- [x] Flatten canonical artifacts to `artifacts/<artifact-id>.ckpt` only when hard-link support is confirmed.
- [x] Build each artifact in a hidden sibling scratch location.
- [x] Move the completed checkpoint to a hidden sibling file.
- [x] Remove scratch before creating any canonical artifact path.
- [x] Publish the canonical artifact with `os.link()` so an occupied target fails without overwrite.
- [x] Remove the hidden completed file after successful publication; make that cleanup non-fatal after the canonical link exists.
- [x] Apply the same hidden-sibling, create-only hard-link ordering to `studies/<study-id>.json`.
- [x] Retain publication-time Study identity, method/index, exact roster, and one-trial checks.
- [x] Use neither `os.replace()` nor ordinary rename as the canonical flat-file publication primitive.
- [x] Update every artifact path owner and string-literal test path.
- [x] Add an early exporter `output_directory.exists()` rejection before twelve lowerings.
- [x] Retain the late exporter collision check immediately before publication.
- [x] Retain exporter scratch cleanup after failure.
- [x] Retain complete-roster, unique-artifact, chain, horizon, parity, forced XNNPACK delegation, and collision coverage.
- [x] Update ADR 0006 and the current FABLE manual with the exact ordering and no-clobber contract.

### Implementer checks

- [x] Run focused Artifact and Study publication tests.
- [x] Run focused exporter collision and cleanup tests.
- [x] Run `uv run --frozen pytest -q`.
- [x] Run Ruff and Pyright on changed Python files.
- [x] Run `uv run --frozen vulture` and manually classify any finding.
- [x] Run `git diff --check`.

### Reviewer acceptance

- [ ] Reproduce occupied-target preservation for Artifact, Study, and exporter output.
- [ ] Confirm no post-publication cleanup failure can retract a published canonical object.
- [ ] Confirm no compatibility layout or overwrite fallback was added.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 2 — Python interfaces, CLI, and orchestration ownership

**Goal:** collapse shallow Python interfaces while preserving raw durable boundaries and one clear Study/CLI owner.

### Records and loading

- [x] Add `src/fable/records.py` with `StrictFrozenRecord`.
- [x] Migrate only the eight identical strict, frozen, extra-forbid Pydantic bases.
- [x] Keep the deliberately non-strict frozen base in `config.py`.
- [x] Add public `load_study(storage_root, study_id)` in `study.py`.
- [x] Keep a private path-based Study loader for scratch publication.
- [x] Route all five canonical Study consumers through the public loader.
- [x] Keep strict JSON validation and embedded-ID equality with both the path and requested UUID.

### CLI and orchestration

- [x] Flatten `src/fable/cli/` into `src/fable/cli.py`.
- [x] Point `[project.scripts]` directly at `fable.cli:app`.
- [x] Delete the wrapper `main()` and old CLI modules. Add no import shim.
- [x] Update direct imports and focused CLI tests.
- [x] Keep `tuning.py` as the Study-orchestration owner.
- [x] Rename `modeling._run_candidate` to public `fit_candidate`.
- [x] Expose `candidate_scratch_directory()` from `study.py`; do not import private `_study_scratch` across modules.
- [x] Give Study scratch paths one owner.
- [x] Inline `_require_method()` into `Study.validate_methods()`.
- [x] Inline one-use `_Objective`, modeling `_NonNegativeInt`, feature-state width, and selected-epoch expressions at their owning fields/construction.
- [x] Add `_CandidateAssociation.training_definition`.
- [x] Delete `_training_definition()` and consume `association.training_definition`.
- [x] Keep `_hydrate_association()` and `_json_association()` as serialization-boundary owners.

### Remote workflow and public names

- [x] Add `BaselineSource.experiment`.
- [x] Make both training-source variants expose `source.experiment`.
- [x] Delete the `isinstance(source, BaselineSource)` branch and remote-CLI BaselineSource import.
- [x] Delete the baseline parameter and unused large fixture from the remote-workflow test.
- [x] Keep selected-training and evaluation remote-workflow coverage.
- [x] Delete leaf-module `__all__` lists.
- [x] Keep package re-exports only where consumed.
- [x] Express retained re-exports as explicit aliases.
- [x] Leave `temporal/__init__.py` docstring-only.

### Owned test changes

- [x] Keep only tuning candidate index `1` in the indexed-result test.
- [x] Keep only the later retention failure case for scratch-preservation evidence.
- [x] Rename test patches from `_run_candidate` to `fit_candidate`.
- [x] Preserve publication-time Study rejection coverage.

### Implementer checks

- [x] Run focused records, Study, tuning, CLI, and remote-workflow tests.
- [x] Run `uv run --frozen pytest -q`.
- [x] Run Ruff, Pyright, Vulture, and `git diff --check`.

### Reviewer acceptance

- [ ] Verify all five Study consumers use the public loader.
- [ ] Verify entry-point installation and CLI help/import behavior.
- [ ] Verify no old module path, shim, silent filter, or private cross-module import survives.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 3 — Scientific core, evaluation, and numerical runtime

**Goal:** remove redundant internal computation while retaining causal, scientific, numerical, and native-runtime boundaries.

### Runtime policy

- [x] Add `_runtime.configure_torch()` as the sole owner of numerical runtime policy.
- [x] Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before GPU work.
- [x] Configure deterministic algorithms, cuDNN determinism/benchmarking, float32 matmul precision, and CUDA/cuDNN TF32 there.
- [x] Call it before training and evaluation GPU work.
- [x] Source Lightning `deterministic` and `benchmark` arguments from the same constants.
- [x] Extend the existing evaluation-policy test with the environment variable. Add no dedicated policy test.
- [x] Run no GPU job. Confirm the policy during the next separately authorized L40 run.

### Temporal and scientific mechanics

- [x] Keep `BlockFrame` and `BlockFrame.select_range()` as validation/range owners.
- [x] Keep its internal `object.__new__` path for validated slices.
- [x] Delete `_require_complete_support()` and duplicate calls.
- [x] Keep testing-window leakage validation, outcome chunking, and observation preallocation.
- [x] Convert `_feature_values()` to direct `match` dispatch.
- [x] Delete `_forming_base_fee_log()`.
- [x] Build named forming-fee columns, combine them with `zip(*columns, strict=True)`, and pass rows directly to `_forming_child_base_fee()`.
- [x] Keep `Series.to_list()` and `_forming_child_base_fee()` so EIP-1559 arithmetic remains arbitrary-precision and exact.
- [x] Replace redundant copied NumPy wrappers with Polars `to_numpy(writable=True)` for schema-owned Int64 columns.
- [x] Replace `total.sum() / batch_size` with `total.mean()`.
- [x] Remove the no-op dtype cast from `outcomes.argmin(axis=1)`.
- [x] Keep indexed minimum gather; do not add a second `min(axis=1)` scan.

### Evaluation and native plumbing

- [x] Derive observation allocation from `OBSERVATION_SCHEMA`.
- [x] Remove evaluation dtype coercions guaranteed by dataset/model contracts.
- [x] Collapse evaluation loading so it does not return a discarded request.
- [x] Keep evaluation ID/path equality.
- [x] Let Lightning capture the incoming association dictionary directly in `save_hyperparameters(logger=False)`.
- [x] Remove the serialize–hydrate–serialize round trip.
- [x] Remove derived evaluation/export values stored only to be discarded.
- [x] Remove exporter host shape, dtype, and finite prechecks duplicated by parity and semantic checks.
- [x] Retain native bridge type/arity, semantic output, parity, and XNNPACK delegation checks.
- [x] Remove `_report_sizes()` and its call.
- [x] Keep `validation_total_loss` as the sole per-epoch Slurm progress value.

### Owned test changes

- [x] Collapse the Python feature fit/transform test to the comprehensive seven-feature forming case.
- [x] Delete strict-subset activity/hour cases and their scaffolding.
- [x] Keep fitted-state, ordering, float32, contiguity, held-out, priority-fee, interval, and predecessor-alignment evidence.
- [x] Keep only the schema-order mutation among equivalent whole-schema BlockFrame rejections.
- [x] Delete the basic isolation test covered by range-selection source/return isolation.
- [x] Delete Corpus priority-fee and seven-column Parquet cases that repeat BlockFrame validation.
- [x] Keep corrupt Parquet, JSON, UUID, and anchor boundary cases.
- [x] Keep constant-feature and constant-target-state rejection tests.
- [x] Delete the loader-profile constant-mirroring test; real CPU training tests exercise the zero-worker path.

### Implementer checks

- [x] Run focused modeling, temporal, corpus, evaluation, and exporter tests.
- [x] Run `uv run --frozen pytest -q`.
- [x] Run Ruff, Pyright, Vulture, and `git diff --check`.

### Reviewer acceptance

- [ ] Trace every removed guard to its upstream typed/scientific owner or surviving final check.
- [ ] Confirm EIP-1559 integer arithmetic, causal geometry, zero-variance rejection, feature finiteness, gas-utilization semantics, and native output checks remain.
- [ ] Confirm runtime configuration occurs before any GPU work without starting a GPU job.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 4 — Experiment runners and mobile exporter contract

**Goal:** give experiment bundles and the exporter direct typed ownership with one frozen workflow.

### Experiment identities and bundles

- [x] Remove `--experiment-id` from all five experiment `prepare` commands.
- [x] Mint every new experiment ID with `uuid4()` and print it.
- [x] Delete the five unreachable UUID-version guards and `tests/experiments/test_cli.py`.
- [x] Make tests parse the minted ID from stdout.
- [x] Keep upstream experiment IDs explicit.
- [x] Add `experiments/bundle.py`.
- [x] Own paths with `bundle_path(storage_root, kind, id)`.
- [x] Share only generic `write_cells(bundle, header, rows)` and `read_cells(bundle)`.
- [x] Rename HPO `candidates.tsv` to `cells.tsv`.
- [x] Keep runner row schema, header, scientific semantics, and selection local.
- [x] Use `enumerate(product(...))` for Cartesian HPO and feature-ablation cells.
- [x] Keep grouped c/k-study loops and derive indices from `len(rows)`.
- [x] Delete mutable counters.

### Experiment contracts

- [x] Delete only the duplicate read-time c-study trial-count check.
- [x] Keep authoring-time feature-ablation and context-study completeness.
- [x] Require exact ordered HPO retained-method equality with `request.methods`.
- [x] Add `ExperimentEntry.require_artifact_id()`, `require_study_id()`, and `require_evaluation_id()`.
- [x] Replace scattered `None` guards and silent filters with those accessors.
- [x] Keep feature-ablation and HPO Methods independent.

### Mobile roster and manifest

- [x] Replace exporter `_Roster`/`_RosterChain` models with `dict[str, dict[int, UUID4]]`.
- [x] Change `MOBILE.yaml` to `chain -> integer horizon -> artifact UUID`.
- [x] At YAML hydration, require exactly three chains, horizons 2–5, UUIDv4 values, and twelve unique IDs.
- [x] Use direct indexing after validation.
- [x] Keep artifact/corpus chain and artifact/horizon validation.
- [x] Remove `_Cell.chain_id`; derive expected IDs from `_CHAINS`.
- [x] Remove emitted `chain_id` and `executorch_version`, their TypeScript fields, constants, fixtures, tests, and docs.
- [x] Delete `_require_versions`; trust the committed isolated project and lockfile.
- [x] Record the used export version only in deferred final-export evidence.
- [x] Keep one frozen exporter environment and supported execution workflow.
- [x] Remove inert exporter Pyright suppressions.

### Owned tests

- [x] Consolidate chain/horizon mismatch cases without losing postconditions.
- [x] Use one module-level deterministic `TinyModel` seed.
- [x] Keep incomplete roster, duplicate artifact, early/late collision, scratch cleanup, genuine export, forced XNNPACK, host execution, and parity coverage.
- [x] Delete `test_experiment_kinds_map_to_their_manifest_namespaces`; runner tests pin all five canonical manifest paths.
- [x] Update app manifest fixtures for the clean schema.

### Implementer checks

- [x] Run all experiment tests.
- [x] Run `uv run --project tools/mobile-export --frozen pytest -q`.
- [x] Run `uv run --frozen pytest -q`.
- [x] Run `npm test` and `npm run typecheck` in `app/`.
- [x] Run Ruff, Pyright, Vulture, and `git diff --check`.

### Reviewer acceptance

- [ ] Verify experiment IDs are minted, printed, and consumed through the real workflow.
- [ ] Verify the exporter raw boundary rejects malformed/incomplete/duplicate rosters.
- [ ] Verify no runtime version parser or second environment survives.
- [ ] Verify app/exporter manifest types agree exactly.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 5 — App engine, model, and history

**Goal:** collapse app lifecycle and internal validation while preserving stale-work, raw RPC bigint, numerical, and native-model boundaries.

### Engine and serialization

- [x] Delete `app/src/engineLifecycle.ts` and its tests.
- [x] Let the chain-selection effect own engine creation, polling stop, and cleanup.
- [x] Keep model serialization, RPC cancellation, engine revisions, and selection revisions.
- [x] Accept brief overlap between independent model loads during rapid chain changes.
- [x] Delete the root mounted ref, mounted guards, unreachable lifecycle catches, and `onRpcUnavailable`.
- [x] Let `onStatus("offline")` clear the snapshot.
- [x] Add local `app/src/serialQueue.ts` with `createSerialQueue()`.
- [x] Use it for model operations, RPC synchronization, and history load/write serialization.
- [x] Add neither `async-mutex` nor a queue-helper choreography test.

### Preparation and failures

- [x] Inline `prepareSelection()` into `prepare()`.
- [x] Keep `Promise.allSettled()` so readiness waits for model and chain settlement.
- [x] Add the second microtask to the focused preparation test.
- [x] Collapse duplicate preparation settlement handlers into one local closure.
- [x] Add `attempt(message, work)` for public inference failure translation.
- [x] Delete `ModelOutputError`.
- [x] Return decoded native outputs directly from `model.execute()`.
- [x] Wrap execution and prediction decoding in one run failure.
- [x] Keep underlying causes and selection-revision checks.
- [x] Add a local `fail(message)` state-transition helper inside `App()`.
- [x] Keep stale-revision/AbortError decisions at call sites.
- [x] Add no reducer, hook, module, or helper test.

### Trusted history and dead data

- [x] Parse `fable.runs` directly as `InferenceRun[]` after JSON parsing.
- [x] Delete custom record validation and its tests.
- [x] Remove all history/display caps and show every matching run.
- [x] Inline `recordOutcome()` and remove its already-resolved guard.
- [x] Remove `ChainSnapshot.chain`.
- [x] Remove stored `InferenceResult.immediate_block`; derive it from `head_block + 1`.
- [x] Remove result/run `head_base_fee_per_gas`; retain it only for RPC polling and model input.
- [x] Remove other derived values without production consumers.
- [x] Clear existing development `fable.runs` when applying the schema clean break. Add no migration.
- [x] Remove `hourAngle()`’s negative-timestamp guard and `positiveLog()`’s positivity guard.
- [x] Keep final feature finiteness and gas-utilization semantics.

### Owned tests

- [x] Keep one unsafe external head-block bigint inference rejection.
- [x] Delete obsolete unsafe head-base-fee and immediate-block cases.
- [x] Keep unsafe outcome-fee and feature-input cases.
- [x] Keep a complete K2–K5 inference manifest; use a model-entry helper, not partial casts.
- [x] Keep the preparation wait/retry test with its second microtask.
- [x] Keep one chain-read, model-load, and run-failure assertion.
- [x] Delete separate malformed-output taxonomy coverage.
- [x] Keep nonfinite decoded-prediction rejection.
- [x] Delete standalone P50 row-alignment coverage already owned by the oracle fixture.
- [x] Delete the interval half of the arithmetic test; keep exact forming-fee integer cases.
- [x] Replace broad model `bundle()` use with one complete catalog fixture and one direct `selection()` helper.
- [x] Delete the import-time `initExecutorch()` call-count assertion.
- [x] Keep model serialization, replacement, disposal, retry, and native-output decoding tests.

### Implementer checks

- [x] Run focused inference, model, feature, history, and App tests.
- [x] Run `npm test` and `npm run typecheck` in `app/`.
- [x] Run `git diff --check`.

### Reviewer acceptance

- [ ] Probe selection change, disposal, failed preparation retry, serial execution, and history persistence.
- [ ] Confirm raw bigint, numerical prediction, feature finiteness, and native decoder guards remain.
- [ ] Confirm no lifecycle abstraction, history cap, migration, or error taxonomy survives.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 6 — App RPC and analytics

**Goal:** remove duplicate batching/analytics work and compress RPC tests without weakening real network behavior.

### RPC production

- [x] Make `features.ts` the sole owner of priority-fee and interval feature constants.
- [x] Import them into `rpc.ts`.
- [x] Type ordered features as `readonly FeatureName[]`.
- [x] Remove manual sequential 40-call chunking.
- [x] Fetch the exact requested range directly and let Viem packetize with `batchSize: 40`.
- [x] Delete the maximum-active-call test.
- [x] Always read both logical outcome blocks through `Promise.all()`.
- [x] Delete action-zero provider-call-count evidence; retain returned values.
- [x] Inline `validateLinks()` while keeping `findBrokenLink()`.
- [x] Keep chain verification, exact block identity, hashes, base fee, bigint shape, fee-history coverage, reorg recovery, abort, timeout, and stale-result checks.

### Analytics production

- [x] Keep only the positive immediate-fee denominator check for app-produced outcomes.
- [x] Delete finite and selected-negative checks.
- [x] Delete `formatGwei()`’s invalid-number fallback.
- [x] Rewrite `summarizeRuns()` as one direct loop computing each realized saving once.
- [x] Keep incremental means because history is unbounded.

### RPC test consolidation

- [x] Add one narrow `fakeChain()` in `app/test/rpc.test.ts`.
- [x] Give it a default chain ID, mutable head, block/fee-history responses, and request/read log.
- [x] Permit small overrides and reject unknown methods.
- [x] Use it only for ordinary deterministic tests.
- [x] Keep bespoke synchronization-race, slow-polling, disposal, production-HTTP abort, and timeout providers local.
- [x] Assert warm synchronization verifies the chain once.
- [x] Delete the redundant `Set.size` assertion after exact ordered block equality.
- [x] Keep the action-zero test only as compact outcome-value evidence.
- [x] Trim the local production JSON-RPC responder while proving distinct session signals, isolated disposal abort, and replacement completion.
- [x] Keep the production 10-second timeout test and stale synchronization-after-disposal test.

### Analytics test consolidation

- [x] Delete the standalone selection/count test; the chart test covers both filters.
- [x] Delete the invalid-fee test after the trusted-history clean break.
- [x] Keep missing-outcome, zero-denominator, loss, zero-selected-fee, chart, and empty-selection behavior through the remaining tests.

### Implementer checks

- [x] Run focused RPC, analytics, history, and inference tests.
- [x] Run `npm test` and `npm run typecheck` in `app/`.
- [x] Run `git diff --check`.

### Reviewer acceptance

- [ ] Probe warm verification, concurrent sync, reorg recovery, disposal cancellation isolation, and the exact timeout boundary.
- [ ] Confirm Viem batching cannot truncate the requested range.
- [ ] Confirm remaining analytics tests cover every supported outcome state.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 7 — App presentation and shared styles

**Goal:** collapse duplicated screen markup/styles and let charts own their natural layout.

### Shared components and styles

- [x] Add `DetailRow.tsx` with `label`, formatted string `value`, and optional `last`.
- [x] Replace Analytics’ local detail component and Inference’s open-coded rows.
- [x] Keep conditional formatting in screens; let `DetailRow` own row markup and final border only.
- [x] Add no `DetailList` or component test.
- [x] Create one `app/src/styles.ts`.
- [x] Move both screen StyleSheets and Horizon slider styles there.
- [x] Normalize page, title, section, surface, network-card, button, dialog, detail-row, label, and value roles.
- [x] Keep unique graph, timeline, and prediction geometry under distinct names in the same file.
- [x] Keep colors in `theme.ts`; remove screen color literals.
- [x] Use direct composed styles without override chains or wrapper components.
- [x] Accept small visual normalization where duplicates differ.
- [x] Remove `NetworkIcon.color`, its fallbacks, and the `graphs` alias.

### Analytics layout

- [x] Remove the outer horizontal carousel.
- [x] Render the three graph cards vertically.
- [x] Let Gifted Bar Chart own internal horizontal scrolling.
- [x] Remove `adjustToWidth`, window/layout width state, carousel state/ref, snapping, momentum, pagination, explicit width, and derived bar geometry.
- [x] Remove default-valued props and duplicate transparent-axis settings.
- [x] Keep semantic dark colors, fee-pair colors/spacing, unit labels, negative-label placement, and bounded negative scale.
- [x] Replace rotated axes with card titles:
  - `Recommended wait distribution`
  - `Savings by wait (%)`
  - `Base fee by wait (Gwei)`
- [x] Split chart algorithms into named private components.
- [x] Add no `ChartFrame`.

### Horizon and dates

- [x] Keep Analytics horizon independent and local.
- [x] Optionally rename `graphHorizon` to `analyticsHorizon`.
- [x] Return `HorizonSlider` without its inert wrapper.
- [x] Derive min/max from `HORIZONS` for bounds and accessibility metadata.
- [x] Keep slider accessibility, colors, step, and typed callback local to the component.
- [x] Replace manual date formatting with one module-level `Intl.DateTimeFormat`.
- [x] Use `en-GB`, numeric day, short month, two-digit hour/minute, and `hourCycle: "h23"`.

### Implementer checks

- [x] Run `npm test` and `npm run typecheck` in `app/`.
- [x] Run Expo Doctor in `app/`.
- [x] Search screens for remaining local StyleSheets, raw color literals, carousel/width state, and deleted props.
- [x] Run `git diff --check`.

### Reviewer acceptance

- [ ] Inspect both screens’ rendered ownership and chart props.
- [ ] Confirm width/carousel state is gone and internal chart scrolling remains.
- [ ] Confirm all shared colors/styles have one owner without override machinery.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 8 — Cross-cutting test and package-tool consolidation

**Goal:** remove remaining mechanical duplication only after production names and seams are stable.

### Shared Python test mechanics

- [ ] Add plain functions in `tests/helpers.py`; add no fixture or god helper.
- [ ] Share only exact duplication:
  - subprocess `_run`
  - TSV `_rows`
  - `REMOTE_YAML` and `write_remote`
  - three identical dispatch builders
  - three identical `window(first)` builders
  - the duplicate Method in `test_modeling.py`
- [ ] Keep scientific Methods, Corpus/block builders, golden arrays, and publication algorithms local.
- [ ] Import through `tests.helpers`.
- [ ] Add no pytest `pythonpath` setting.

### Package and static configuration

- [ ] Remove unused SQLite-journal, Mypy, and coverage `.gitignore` entries.
- [ ] Remove explicit Hatch wheel include/source configuration for conventional `src/fable`.
- [ ] Extend Vulture paths to `src`, `tests`, `experiments`, and `tools/mobile-export`.
- [ ] Build the wheel once and verify it contains and imports `fable`.
- [ ] Delete only tests made obsolete by approved production deletions.
- [ ] Keep one focused rejection test per live raw parser.
- [ ] Keep scientific, numerical, atomic-publication, native-runtime, and demonstrated-failure tests.
- [ ] Add no compatibility, transition, private-call-count, or library-choreography test.

### Implementer checks

- [ ] Run `uv build`.
- [ ] Inspect wheel contents and import `fable` from the built wheel in an isolated temporary environment.
- [ ] Run `uv run --frozen pytest -q`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run pyright`.
- [ ] Run `uv run vulture` and manually validate every finding.
- [ ] Run `git diff --check`.

### Reviewer acceptance

- [ ] Confirm each helper removes exact structure without coupling scientific fixtures.
- [ ] Confirm the wheel contains/imports the package without custom Hatch mapping.
- [ ] Manually audit Vulture findings against callbacks, validators, CLI registration, reflection, and configuration.
- [ ] Issue separate Standards and Spec verdicts.

## Slice 9 — Documentation and repository layout

**Goal:** leave one current documentation tree with correct links and no stale contract text.

### Layout and links

- [ ] Move `FABLE.md` to `docs/FABLE.md`.
- [ ] Move `CONTEXT.md` to `docs/CONTEXT.md`.
- [ ] Keep only `README.md` and discovery-critical `AGENTS.md` as root Markdown files.
- [ ] Keep every other Markdown file under `docs/`.
- [ ] Rewrite every inbound/relative link, including research links.
- [ ] Preserve README-linked headings and anchors.

### Manual and glossary

- [ ] Compress the manual by about 97 lines as an estimate, not quota.
- [ ] Preserve its standalone scientific contract, equations, causal rules, estimands, feature/target definitions, claim boundaries, limitations, sources, provenance, and ownership statements.
- [ ] Remove only proven repetition, stale implementation wording, and mobile details owned elsewhere.
- [ ] Delete nonexistent `apply_method()` wording and describe direct Method membership plus `TrainingDefinition`.
- [ ] Add public `reduce_rolling()` to the evaluation API.
- [ ] Describe hydrate-once strict records and trusted nested typed values.
- [ ] State that only `model_width` must be even/divisible by `attention_heads`.
- [ ] Regenerate the dependency diagram from imports, include `tuning`, and correct `modeling`/`study`.
- [ ] State UUID syntax for `study finalize` and UUIDv4 origin for publishable TuneRequests.
- [ ] Delete the inapplicable macro-F1 zero-division claim.
- [ ] Remove the twelve glossary entries that merely restate Pydantic records.
- [ ] Do not add “Cost over optimum.”

### ADR, agent, and research cleanup

- [ ] Delete ADRs 0001–0005.
- [ ] Add `docs/adr/README.md` listing number, title, final status, and successor.
- [ ] Keep ADRs 0006 and 0007 in full.
- [ ] Delete `docs/agents/triage-labels.md`.
- [ ] List only `ready-for-agent`, `ready-for-human`, and `wontfix` as ordinary triage labels in `AGENTS.md`.
- [ ] Leave Wayfinder labels/behavior unchanged.
- [ ] Delete `docs/agents/domain.md`.
- [ ] Fold only terminology-from-`docs/CONTEXT.md` and ADR-conflict rules into `AGENTS.md`.
- [ ] Remove the PR-triage section from `docs/agents/issue-tracker.md`.
- [ ] Collapse duplicate README navigation/architecture rows and keep one ADR pointer.
- [ ] Delete tracked orphan research assets:
  - `docs/research/evm-fees.html`
  - `docs/research/gas-base-fees-and-priority-fees.html`
  - `docs/research/teaching.css`
- [ ] Keep `priority-fees.md` and `on-device-inference.md`.
- [ ] Leave the untracked GPU research note untouched.

### Implementer checks

- [ ] Search the repository for old document paths, deleted ADR names, deleted APIs, old artifact paths, removed manifest fields, caps, and stale links.
- [ ] Verify every local Markdown link target and README anchor.
- [ ] Run `uv run --frozen pytest -q`.
- [ ] Run `uv run ruff check .`, `uv run pyright`, and `uv run vulture`.
- [ ] Run `uv run --project tools/mobile-export --frozen pytest -q`.
- [ ] Run `npm test`, `npm run typecheck`, and Expo Doctor in `app/`.
- [ ] Run `git diff --check`.

### Final reviewer acceptance

- [ ] Review the documentation slice against the implemented repository, not the pre-implementation plan.
- [ ] Run the full cumulative Python, exporter, app, type, static, wheel/import, link, and diff matrix.
- [ ] Confirm every prior slice commit is reachable in order and each prior authoritative review was GREEN.
- [ ] Confirm the worktree/index contain only the intentional untracked GPU note.
- [ ] Issue separate Standards and Spec verdicts and final `GREEN LIGHT`.

## Deferred final-model export and exporter retirement

This work is not Slice 10. It requires selected real artifacts and native execution evidence unavailable during this program. Its sole owner is [A01 — Activate selected artifacts in the final Mac/mobile demo](https://github.com/edoski/fable/issues/139), now updated with:

- the final twelve artifact selections;
- frozen exporter execution;
- twelve `.pte` files and manifest;
- roster, collision, parity, XNNPACK, and cleanup gates;
- all-twelve-cell native simulator comparison;
- one representative physical-phone outcome;
- one-time provenance evidence;
- exporter/tool retirement only after every gate passes.

Do not delete `tools/`, `MOBILE.yaml`, exporter tests, its isolated environment, or exporter documentation in this simplification program.

## Explicitly rejected

- Merge `tuning.py` into `modeling.py`.
- Convert `BlockFrame` into a dataclass plus validating factory.
- Couple Analytics horizon to inference horizon.
- Replace frozen experiment Methods with shared mutable definitions.
- Add `async-mutex`, schema adapters, compatibility readers, history caps, runtime manifest revalidation, or alternate export workflows.
- Remove final atomic collision checks, raw-input checks, scientific completeness, causal/leakage checks, feature finiteness, gas-utilization semantics, native bridge semantics, parity, or XNNPACK delegation proof.
- Run experiments, live RPC acquisition, Slurm jobs, model export, deployment, or final asset publication inside these nine slices.

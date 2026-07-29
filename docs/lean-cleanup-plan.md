# Lean cleanup implementation plan

Status: approved plan

Target: clean `main` at `2ea05e0468d6748864d285b8ef5e10c09b5e4ddf`

## Goal and operating assumptions

Make FABLE smaller and easier to read for a bounded undergraduate thesis project. Prefer the
direct successful path, local ownership, and few concepts. This is a clean break: do not add
compatibility aliases, fallback readers, migrations, transition tests, or generalized
registries.

The supported operator path is intentionally narrow:

- Experiment scripts or Codex-authored typed code create every request and bundle. This includes
  later add-on GPU jobs that extend an experiment.
- The user does not manually construct request JSON or edit bundle rows.
- Generated requests are persisted before submission and contain their minted UUIDv4 IDs.
- A later extension creates a new durable destination. Existing Study and artifact IDs may be
  reused only as explicit source references. Resuming the same work reuses its complete persisted
  request.
- Existing raw-data, scientific, numerical, no-overwrite publication, resume, and native-runtime
  protections remain unless a slice explicitly replaces them.
- Do not import the compact-CUDA branch's temporal or runtime changes.
- Do not run experiments, submit Slurm jobs, alter queued work, or update the university checkout.

## Revised decisions

### Request construction

Delete `fable.requests`. Put literal workflow defaults and UUIDv4 default factories directly on
`TrainRequest`, `TuneRequest`, and `EvaluateRequest`.

This accepts that a manually authored incomplete concrete request could mint a missing ID during
model hydration. Manual request authoring is outside the supported path. Experiment scripts and
Codex-authored utilities construct models directly, then persist complete JSON containing the
minted identity. This remains suitable for later experiment extensions: a new Tune, Train, or
Evaluate destination receives a new ID, while its existing Corpus, Study, or artifact source ID is
passed explicitly. Canonical loaders still compare embedded IDs with requested paths.

Do not create a new work unit by copying an existing request while retaining its destination ID.
Construct a new request, or explicitly replace the destination ID. Reuse the complete original
request only when resuming the same logical work.

Concretely, later HPO values create a new `TuneRequest` and Study ID; a later fit creates a new
`TrainRequest` and artifact ID; and a later evaluation creates a new `EvaluateRequest` and
evaluation ID. Each new request may point to existing Corpus, Study, or artifact inputs. Completed
canonical records are referenced, not extended in place.

### Packed workflow uniqueness

Keep `_workflow_identity` and destination-based duplicate rejection.

Later Codex-authored extensions make same-destination/different-payload requests realistic enough
to guard: copying a prior request and changing its source, Method, or evaluation window can retain
the old destination ID. Two such requests compare unequal as complete Pydantic values but still
write the same artifact or evaluation path. The existing seven-line helper prevents that pair from
sharing one packed allocation without adding a new abstraction. Candidate submission keeps the
equivalent `(study_id, method_index)` slot check.

## Slice 1 — Typed spine and generated request construction

### Implementer work

1. In `src/fable/config.py`:
   - Import `uuid4`.
   - Give `TrainRequest.workflow`, `TuneRequest.workflow`, and `EvaluateRequest.workflow` their
     literal defaults.
   - Give `artifact_id`, `study_id`, and `evaluation_id` UUIDv4 default factories at their owning
     request models.
   - Keep source IDs such as Corpus, selected Study, and evaluated artifact IDs required.
2. Replace every `fresh_train_request`, `fresh_tune_request`, and `fresh_evaluate_request` call
   with direct construction of the corresponding request model.
3. Delete `src/fable/requests.py`.
4. Update request tests to omit generated fields where construction should mint them. Existing
   experiment tests must continue to assert UUID version 4 and per-cell uniqueness.
5. In `src/fable/_runtime.py`, inline the one-use float32 matmul and TF32 values inside
   `configure_torch()`. Keep batch-size and DataLoader tuning constants.
6. Update `docs/FABLE.md` so `fable.config`, not `fable.requests`, owns fresh request construction.
   Remove the fresh-constructor reference section.
7. Keep `records.py`, `addresses.py`, the strict request union, and all three concrete model
   definition records.

### Expected outcome/state

- `src/fable/requests.py` no longer exists.
- `rg "fresh_(train|tune|evaluate)_request|fable\\.requests" .` finds no live references.
- Direct `TrainRequest`, `TuneRequest`, and `EvaluateRequest` construction mints exactly one
  destination UUIDv4 when its generated ID is omitted.
- Persisted JSON includes the workflow discriminator and minted ID.
- Parsing generated request JSON does not mint a replacement identity.
- Corpus IDs, selected Study IDs, source artifact IDs, and other association IDs remain required.
- Later experiment extensions can be generated without the original experiment script: new work
  is constructed as a new typed request, then persisted before submission.
- Extending completed data creates a new Study, artifact, or evaluation destination. It does not
  mutate an existing canonical output.
- Runtime behavior is unchanged apart from removing names for three fixed one-use values.

### Focused verification

- `uv run pytest tests/test_config.py tests/experiments tests/cli`
- `uv run ruff check src/fable/config.py src/fable/_runtime.py experiments tests`
- `uv run pyright`

## Slice 2 — Modeling and Study publication

### Implementer work

1. Replace `_CandidateAssociation` with direct `TrainingDefinition` input for candidate fits.
   Keep `ArtifactAssociation` unchanged for canonical artifacts.
2. Adjust the private association adapter/helper so `_FitModule` obtains one
   `TrainingDefinition` from either:
   - an `ArtifactAssociation`; or
   - a candidate `TrainingDefinition`.
3. Preserve candidate-specific scratch directories and full-state `last.ckpt` resume.
4. Simplify artifact publication:
   - Hardlink the selected best checkpoint directly from scratch to the canonical artifact path.
   - Let the hardlink enforce no overwrite.
   - Remove scratch after the canonical link succeeds.
5. Simplify Study publication:
   - Assemble the completed Study inside its scratch directory.
   - Hardlink it directly to the canonical Study path.
   - Remove scratch after the canonical link succeeds.
6. Delete hidden-sibling cleanup and best-effort unlink code.
7. Delete hidden-cleanup-failure tests. Retain one focused collision/no-clobber test for each
   materially different publication path.
8. Remove verified Lightning defaults only:
   - Keep weighted epoch aggregation and `batch_size`.
   - Keep explicit training epoch logging behavior.
   - Keep finite validation failure, checkpoint filename stability, periodic validation,
     full-state resume, and weights-only best checkpoints.
9. Remove assertions that only restate `self.log` keyword arguments. Keep tests proving short-batch
   weighting and validation base-fee optimality-gap calculation.
10. Update publication and association descriptions in `docs/FABLE.md`.
11. Update `docs/adr/0006-direct-durable-object-authority.md` so its accepted publication
    decision describes direct scratch-to-canonical hardlinking and the accepted leftover-scratch
    tradeoff. Do not leave the ADR describing deleted hidden-sibling cleanup.

### Expected outcome/state

- Candidate checkpoints contain only the training definition required to rebuild the candidate
  model; they do not carry a second request-association record.
- Canonical artifacts still contain their full `ArtifactAssociation`.
- Interrupted candidates resume from their own `last.ckpt`.
- Artifact and Study publication remain atomic and no-replace on the local filesystem.
- A cleanup failure after linking may leave scratch beside a valid canonical output. No
  best-effort cleanup recovery machinery remains.
- Artifact loading still rejects the wrong embedded artifact ID.
- Study loading and selected-Method resolution retain their existing association checks.
- ADR 0006 and `docs/FABLE.md` describe the implemented publication path.

### Focused verification

- `uv run pytest tests/test_modeling.py tests/test_study.py`
- `uv run ruff check src/fable/modeling.py src/fable/study.py tests/test_modeling.py tests/test_study.py`
- `uv run pyright`

## Slice 3 — Evaluation and experiment-manifest cleanup

### Implementer work

1. Remove `deadline_action_k` from:
   - `OBSERVATION_SCHEMA`;
   - observation collection;
   - evaluation fixtures and golden rows;
   - documentation.
2. Keep observation preallocation. Replace the twelve handwritten destination assignments with
   one explicit mapping from schema column name to the current batch array, then fill the
   preallocated arrays in schema order.
3. Replace the unrolled rolling reducer with a descending loop over horizons `5, 4, 3, 2`.
4. Preserve the exact rolling rule: after a horizon `K > 2`, advance an origin only when the
   selected action is terminal, `predicted_action_k == K - 1`.
5. Preserve K=5 as the one-shot observation source and K=2 as the final rolling fee source.
6. Keep the baseline policy tuple and `_economic_metrics` unchanged.
7. Keep `src/fable/experiments.py` in the installed package.
8. Shorten `tests/test_experiments.py`, but retain:
   - manifest write/load round-trip;
   - requested-ID validation through the loader;
   - no-overwrite behavior.

### Expected outcome/state

- Canonical observations have eleven ordered columns and no derivable deadline-action column.
- Observation collection remains bounded by one preallocated array per durable column.
- Evaluation, baseline, and rolling metric values remain unchanged for equivalent observations.
- The rolling reducer expresses one obvious confirmation ladder rather than four copied blocks.
- Experiment manifests remain strict, immutable groupings of canonical record IDs.

### Focused verification

- `uv run pytest tests/evaluation tests/test_experiments.py`
- `uv run ruff check src/fable/evaluation.py src/fable/experiments.py tests/evaluation`
- `uv run pyright`

## Slice 4 — Operator-edge cleanup

### Implementer work

1. In `src/fable/execution.py`:
   - Keep `submit_workflows()` and `submit_candidates()` as the two public typed entry points.
   - Extract their shared tuple/count/remote/render/submit path into one private helper.
   - Keep `_workflow_identity` and reject duplicate workflow destination identities.
   - Keep candidate duplicate rejection based on `(study_id, method_index)`.
   - Keep candidate Method-index validation.
2. Retain one focused workflow test proving that unequal requests with the same destination ID
   fail before SSH submission. Retain the equivalent focused candidate-slot test.
3. Introduce one reusable Typer path annotation that resolves `storage_root` before experiment
   command bodies run.
4. Replace repeated `storage_root = storage_root.resolve()` statements across experiment runners.
5. Move the byte-identical feature-ablation and C-study closure body into
   `experiments/bundle.py`. The helper must:
   - read the bundle rows;
   - load every referenced Study;
   - publish the ordered manifest;
   - remove the temporary bundle only after publication;
   - print the experiment ID.
6. Keep `jobs.tsv` journaling, allocation packing, flushing/fsync, GRES scaling, SSH rendering, and
   Slurm step isolation.
7. Keep current HPO model construction and feature-unit construction. Do not replace them with
   registries or generalized data-driven factories.

### Expected outcome/state

- Submission has one private rendering/invocation path and two small public typed wrappers.
- Requests targeting the same artifact or evaluation destination are rejected before SSH
  submission even when the rest of their payload differs.
- Candidate processes targeting the same Study Method slot are rejected before submission.
- Every supported experiment command receives an already resolved storage root.
- Feature-ablation and C-study share one closure implementation.
- Packed allocations, resume journaling, and generated request contents remain unchanged.

### Focused verification

- `uv run pytest tests/test_execution.py tests/experiments/test_launch.py tests/experiments/test_feature_ablation.py tests/experiments/test_c_study.py`
- `uv run ruff check src/fable/execution.py experiments tests/test_execution.py`
- `uv run pyright`

## Slice 5 — Python test-suite cleanup

### Implementer work

1. Add one focused helper for publishing generated Study fixtures used by experiment-runner tests.
   Keep it specific to TuneRequest rows and retained results.
2. Move the unique C-study assertions into the HPO pipeline test, then delete the standalone
   duplicated C-study subprocess chain.
3. Keep the feature-ablation assertions that define its matrix and the HPO assertions that define
   selection.
4. Keep one explicit canonical observation-schema assertion. Use the production schema when other
   tests only need valid fixture construction.
5. Remove the redundant Polars `schema=` argument where typed NumPy columns already establish the
   same schema.
6. Consolidate repeated valid request, Study, and publication fixtures where this shortens the test
   without hiding the asserted contract.
7. Do not delete tests protecting:
   - temporal/scientific formulas and role boundaries;
   - short-batch metric weighting;
   - candidate interruption and resume;
   - raw workflow discriminator rejection;
   - no-overwrite publication;
   - packed allocation and `jobs.tsv` behavior;
   - native model output shape, dtype, and finiteness.

### Expected outcome/state

- Experiment tests exercise one staged pipeline without re-running the same setup in a standalone
  C-study test.
- Fixture ownership is centralized only where the same construction is repeated.
- Golden schemas have one explicit test owner.
- Removed assertions do not merely reappear behind a generic helper.
- The suite is materially shorter while retaining every scientific, durable, and demonstrated
  failure contract listed above.

### Focused verification

- `uv run pytest tests`
- `uv run vulture`
- `uv run ruff check tests`
- `uv run pyright`

## Slice 6 — Demo-app cleanup

### Implementer work

1. In `AnalyticsScreen.tsx`, extract:
   - a small chart frame owning the graph container and x-axis title;
   - a helper that converts `chartScale()` output into shared Gifted Charts scale props.
2. Keep each chart's data preparation, colors, labels, null handling, negative-axis behavior, and
   grouped-bar spacing in its own component.
3. In `App.tsx`, replace the two-field applied/intended selection ref with one selection value and
   one revision gate.
4. Keep the engine-identity and revision checks that prevent stale inference results from becoming
   visible after a selection change.
5. Keep serialized history persistence and outcome resolution.
6. Remove the initial history-load unmount flag.
7. In `inference.ts`, simplify disposal to dispose the chain session and return model disposal.
   Delete the unused session-error preservation sequence.
8. Add shared typed builders for repeated `InferenceResult` and `InferenceRun` fixtures.
9. Remove the history test that only proves native `JSON.parse` throws `SyntaxError`.
10. Keep the Viem transport configuration test and all RPC/native boundary tests.

### Expected outcome/state

- Three chart components share presentation framing without sharing their distinct data models.
- Selection has one canonical value and one stale-work revision.
- Changing chain or horizon cannot publish an obsolete visible inference result.
- History writes and pending-outcome retries remain serialized.
- Engine disposal remains idempotent through the model runtime.
- App tests contain one canonical typed run/result fixture source.
- No production validation at RPC, numerical, model-output, or native-resource boundaries is
  removed.

### Focused verification

- From `app/`: `npm test -- --run`
- From `app/`: `npm run typecheck`

## Slice 7 — Temporal micro-cleanup

### Implementer work

1. Remove `HistoricalDataset._sample_count`.
2. Remove the `sample_count` constructor parameter.
3. Derive `__len__()` from the stored labels.
4. Update dataset construction and focused tests.
5. Make no other temporal or `BlockFrame` changes.

### Expected outcome/state

- Historical dataset length has one owner: the labels array.
- Training, validation, and testing batch contents and ordering remain unchanged.
- Main's CPU DataLoader architecture remains intact.
- Outcome chunking, reused training outcomes, lazy fixed-context slicing, feature-state fitting,
  and canonical `BlockFrame` validation remain unchanged.

### Focused verification

- `uv run pytest tests/temporal tests/test_modeling.py tests/evaluation/test_evaluate.py`
- `uv run ruff check src/fable/temporal.py tests/temporal`
- `uv run pyright`

## Final integrated gate

After all slices:

1. Search for removed concepts:
   - `fresh_train_request`, `fresh_tune_request`, `fresh_evaluate_request`;
   - `fable.requests`;
   - `_CandidateAssociation`;
   - `deadline_action_k`;
   - hidden artifact/Study publication cleanup.
2. Review the final diff for compatibility code, new indirection, duplicated ownership, unrelated
   edits, and compact-CUDA branch residue.
3. Run:

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run vulture
(cd app && npm test -- --run)
(cd app && npm run typecheck)
```

Expected final state: all checks pass, the worktree contains only the approved cleanup, no
experiments or remote jobs have run, and no live university or queued-work state has changed.

## Post-slice CUDA branch propagation

After all seven slices are green on `main`, propagate the final `main` history to
`codex/compact-cuda-execution`.

### Implementer work

1. Start from the exact reviewed CUDA-branch head and merge the final reviewed `main` head.
2. Resolve overlaps so all applicable cleanup slices are present.
3. Preserve the CUDA branch's device-resident historical backing, batching, loader, and runtime
   behavior. Do not replace it with main's CPU DataLoader implementation.
4. Adapt Slice 7's single-owner dataset-length cleanup to the CUDA backing rather than restoring
   main's temporal implementation wholesale.
5. Keep CUDA-specific documentation and focused tests truthful.
6. Commit the merge/integration. Do not push.

### Expected outcome/state

- `codex/compact-cuda-execution` contains all seven applicable cleanup slices from `main`.
- Its historical batches remain device-resident and its CUDA-specific runtime path remains intact.
- No CUDA-specific commit is dropped or overwritten by a main-side file replacement.
- `main` remains unchanged by branch propagation.

### Focused verification

- Run the full Python gate available on the local host.
- Run the app gate because Slice 6 changes the demo app.
- Compare the final CUDA branch against final `main` and verify every remaining temporal/runtime
  difference belongs to the CUDA backing.

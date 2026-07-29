# Lean cleanup implementation plan

Status: active orchestration plan

Initial code baseline: `f08b707cb44dc06c6abe50a20b6b3fe1fe024496`

Execution branch: `main`

Deferred cleanup: [GitHub issue #147](https://github.com/edoski/fable/issues/147)

This plan converts the approved architecture review into small clean-break changes. The goal is
the shortest direct implementation that preserves FABLE's scientific meaning, canonical durable
objects, atomic publication, external execution boundary, and native mobile contract.

The project is a bounded thesis demonstration. Internal typed values and canonical owners are
trusted. Validation remains only where bad input could silently corrupt scientific results,
publish the wrong durable object, cross an RPC/native/process boundary incorrectly, or produce
nonfinite numerical output.

## Fixed decisions

- No compatibility readers, deprecated fields, migration helpers, transition tests, or dual paths.
- Delete tests for deleted machinery. Do not replace them with tests that prove the old design is
  gone.
- Keep raw Parquet/JSON schema checks, causal window checks, positive fitted standard deviations,
  final feature/target/prediction finiteness, durable request/UUID associations, atomic
  no-clobber publication, Slurm allocation identity, RPC parent continuity and fee-history
  alignment, and ExecuTorch lifetime/parity checks.
- Keep `requests.py`. It centralizes UUID creation and discriminated request construction used by
  several experiment stages.
- Keep `main`'s conventional lazy `HistoricalDataset`. The temporary compact-CUDA design is not
  part of this cleanup. Trim derivable arithmetic state without changing the dataset architecture.
- Keep deterministic training through one seed call and Lightning's deterministic trainer mode.
  Remove duplicate RNG and deterministic configuration machinery.
- Keep `tasks_per_job` because it represents the selected node's usable GPU capacity: two or
  three. Packing must avoid singleton tails when possible. Seven pending fits at capacity three
  must submit as `3 + 2 + 2`, preserving seven concurrent fits in three QoS jobs.
- Keep the rolling policy exactly as specified by the manuscript:
  `K=5 -> K=4 -> K=3 -> K=2`; advance the closed head only when a `K>2` model chooses
  `k=K-1`; every smaller model replaces the previous prediction; `K=2` is final; the original
  five-block deadline never moves.
- Immediate and deadline baselines are economic reference policies, not classifiers. Their result
  rows retain only base-fee savings, P50 fee-inclusive savings, and Cost over optimum.
- Keep one direct app Run path. Model loading and chain synchronization start on Run, not on
  selection. A slower first Run is accepted.
- App RPC synchronization is stateless. Each Run reads the exact required range. Viem owns HTTP
  timeout, retry, batching, and block watching.
- Checkpoint resume stays untouched until every required fit and authorized retry is complete.
  Its later removal is governed by issue #147 and is not part of the active slices below.

### Gradient clipping decision

Delete `_FitModule.configure_gradient_clipping`. `Trainer(gradient_clip_val=...)` already performs
norm clipping. The override only changes PyTorch's nonfinite-gradient behavior from propagation to
an immediate exception.

The lost edge case is a single batch producing NaN or infinite gradients before validation. That
is not an expected operating condition for the fixed architecture, standardized finite inputs,
AdamW, BF16 mixed precision, and configured norm clipping. If it occurs, parameters become
nonfinite and the retained validation-metric finiteness check stops the fit at the epoch boundary.
Losing the exact failing-batch exception is an acceptable diagnostic reduction, not a scientific
behavior change.

### Baseline metric decision

Removing Accuracy and Macro-F1 from immediate/deadline rows makes the code and result contract
smaller:

- no synthetic action arrays for fixed policies;
- no baseline call to `_classification_metrics`;
- two fewer schema fields and corresponding test expectations;
- output matches the manuscript, which defines these policies as economic references and asks for
  their three economic metrics.

Learned-model evaluation still reports Accuracy and Macro-F1.

## Orchestration protocol

Each active slice uses two separate Codex tasks. Both use `gpt-5.6-sol` with `xhigh` reasoning.

1. The orchestrator records `main`'s `HEAD` as the slice baseline.
2. A new implementer task works directly in the saved project checkout on `main`. Its only spec is
   the named slice in this file. It uses the implementation skill, works test-first at useful
   seams, runs focused checks throughout, runs the slice's full checks at the end, and commits its
   work. It does not run a self-review; the separate reviewer owns that gate.
3. After the implementer exits, a new read-only reviewer task works in the same saved checkout. It
   uses the code-review skill
   with `git diff <slice-baseline>...HEAD`, this file as the Spec source, `AGENTS.md`,
   `docs/CONTEXT.md`, and active ADRs as Standards sources. Standards and Spec findings remain
   separate. `GREEN LIGHT` requires zero actionable findings on both axes.
4. On rejection, the orchestrator sends the findings to the same implementer task. The implementer
   fixes and commits. The same reviewer task reviews only
   `git diff <previously-reviewed-head>...HEAD`, plus whether the new hunks close the outstanding
   findings. Stable accepted hunks are not re-reviewed.
5. On `GREEN LIGHT`, the orchestrator runs a proportional integration check, records the result
   below, and starts the next slice from the new `main` `HEAD`.
6. Any scope discovery that changes scientific meaning, durable formats beyond the declared clean
   break, Slurm capacity, or native app behavior returns to the orchestrator. The implementer does
   not broaden the slice.

The plan is a live ledger. After every implementation or review callback, update the task IDs,
baseline and head SHAs, checks, findings, and status before proceeding.

| Slice | Scope | Status | Baseline | Implementer | Reviewer |
| --- | --- | --- | --- | --- | --- |
| 1 | Training core and tuning | GREEN LIGHT | `ffd7c368` | `019fad88-4411-7892-963a-301075fb96f7` / `44435ddf` | `019fad92-f093-75f3-99af-4f93de395d1f` |
| 2 | Experiment manifests, selection, and packed launch | GREEN LIGHT | `1a141fa4` | `019fad98-e122-71c3-b9a5-4a33384fc860` / `2d506cf6` | `019fada3-ec16-7552-becb-6fdaf9a819e3` |
| 3 | Evaluation and rolling reduction | GREEN LIGHT | `e755fbdd` | `019fadaa-4725-7323-813b-21f8760f16e5` / `836577f6` | `019fadb1-9646-7b31-8700-00d2dd9006be` |
| 4 | App inference and model lifecycle | GREEN LIGHT | `b7e26671` | `019fadb6-2d6f-75e3-b3ed-7059843f6ec1` / `0226dcc4..ea0b664b` | `019fadc0-a253-7a23-998d-7ee44c3471e4` |
| 5 | App RPC and feature input path | GREEN LIGHT | `41216ba4` | `019fade1-2207-7490-bc92-05d28d465848` / `51b7a4d0` | `019fadec-9614-7882-a227-278c006c85e6` |
| 6 | Mobile exporter | ready | pending | pending | pending |
| 7 | Documentation truth pass | blocked by 6 | pending | pending | pending |
| 8 | Compact-CUDA parity | blocked by 7 | pending | pending | pending |
| 9 | Remove checkpoint resume | blocked by issue #147 gate | n/a | not started | not started |

Slice 1 review: Standards 0 findings; Spec 0 findings. Focused tests: 22 passed. Full Python
suite: 117 passed. Ruff, Pyright, Vulture, and `git diff --check`: passed. The reviewer directly
confirmed interrupted-fit resume and publication-failure scratch preservation.

Slice 2 review: Standards 0 findings; Spec 0 findings. Focused tests: 34 passed. Full Python
suite: 126 passed. Ruff, Pyright, Vulture, and `git diff --check`: passed. The reviewer also swept
pending counts 1–20 and confirmed minimal allocation count, capacity bounds, stable totals, and no
avoidable singleton.

Slice 3 review: Standards 0 findings; Spec 0 findings. Evaluation tests: 17 passed. Full Python
suite: 125 passed. Ruff, Pyright, Vulture, and `git diff --check`: passed. The reviewer exhaustively
enumerated all 120 four-model action combinations and confirmed the manuscript recurrence and
fixed deadline.

Slice 4 review: final Standards 0 findings; final Spec 0 findings. The first review found a stale
history publication interval during an awaited save. Correction review rejected fallible rollback
and required FIFO linearization of history and selection. The next correction review found a
latest-selection loss against stale rendered state. Final review confirmed one App-owned
intended/applied selection record, latest-intent coalescing, single-write FIFO history, no rollback,
and closure of both P1s. App tests: 47 passed. Typecheck, Expo Doctor 19/19, and
`git diff --check`: passed.

Slice 5 review: Standards 0 findings; Spec 0 findings. The reviewer checked installed Viem 2.55.8
against the implementation and confirmed default-chain transport selection, batches capped at
forty requests, a transport-owned ten-second timeout, zero retries, watcher error forwarding, and
unwatch disposal. Fresh exact-range reads retain parent continuity, fee-history alignment, exact
integer forming-fee arithmetic, and final Float32 finiteness. App tests: 32 passed. Typecheck, Expo
Doctor 19/19, Vulture, and `git diff --check`: passed. Tests used stubbed fetch; no live RPC was
contacted.

## Slice 1 — Training core and tuning

### Files

- `src/fable/_runtime.py`
- `src/fable/temporal.py`
- `src/fable/min_block_fee.py`
- `src/fable/modeling.py`
- `src/fable/tuning.py` (delete)
- `src/fable/cli.py`
- directly affected Python tests
- the affected training sections of `docs/FABLE.md`

### Implementation

1. Remove the dedicated CPU `torch.Generator` from `_fit`, the `generator` argument from
   `_runtime.data_loader`, and its DataLoader forwarding. Keep
   `pl.seed_everything(fit.seed)` without `workers=True`. Shuffling may now depend on prior global
   Torch RNG consumption within the seeded fit; that small run-to-run ordering difference is
   accepted.
2. Let Lightning own deterministic setup:
   - keep `Trainer(deterministic=True)`;
   - remove `_runtime.DETERMINISTIC`, `_runtime.BENCHMARK`, the `benchmark=` trainer argument, and
     duplicate CUBLAS/cuDNN/deterministic-algorithm setup from `configure_torch`;
   - keep BF16 precision, batch sizes, float32 matmul precision, and TF32 flags.
3. Delete `_FitModule.configure_gradient_clipping` and the explicit
   `gradient_clip_algorithm="norm"` default. Keep `gradient_clip_val` and validation-metric
   finiteness.
4. Make `min_block_fee_loss` return the per-origin total-loss tensor directly. Training owns
   `.mean()` for backpropagation. Validation owns detached float64 accumulation/mean for correctly
   weighted epoch metrics. Delete `MinBlockFeeLoss`.
5. Remove arithmetic tensors from historical storage:
   - `_HistoricalBacking` stores `first_block`, inputs, and base fees only;
   - `HistoricalDataset` stores one `first_origin_row` integer and its sample count instead of a
     contiguous `_origin_rows` tensor;
   - item origin rows derive as `first_origin_row + index`;
   - origin block derives as `backing.first_block + origin_row`;
   - local NumPy origin arrays may remain during chunked outcome construction.
6. For the Ethereum-only exact-forming-fee feature, trust `BlockFrame`'s single-chain definition
   instead of scanning the full `chain_id` column. Pass the authoritative chain ID into feature
   derivation or place the check at the nearest existing owner; do not add a second wrapper.
7. Move `run_candidate` into `modeling.py`, fold or privatize the one-caller candidate-fit helper,
   update the CLI import, and delete `tuning.py`. Preserve the exact transaction:
   fit into candidate scratch, atomically retain the result, then remove candidate scratch.
   Any fit/publication failure must preserve scratch while resume remains active.
8. Remove redundant `strict=True` from `StrictFrozenRecord.model_validate_json` calls in files
   touched by this slice. Do not change standalone `TypeAdapter` behavior without a focused proof.

### Test change

Delete:

- exact dedicated-generator isolation tests;
- the custom clipping-hook test;
- mocked `tuning.py` choreography tests;
- assertions about duplicate runtime flag calls.

Keep or rewrite:

- exact target and loss values, now against the returned per-origin tensor;
- one shuffled loader behavior test without exact RNG sequence;
- causal windows, chunked outcomes, feature parity, finite transforms, dataset device transfer,
  and sample geometry;
- one candidate success transaction and one failure-preserves-scratch test at the new owning seam;
- Transformer initialization, one representative family fit/load path, validation weighting, best
  checkpoint selection, and atomic artifact publication.

### Expected outcome

- One seed owner, one deterministic-runtime owner, one gradient-clipping owner.
- No wrapper object around a single loss tensor.
- No tensor storage spent on derivable block/origin sequences.
- No one-function tuning module.
- Same targets, loss mean, window geometry, artifact format, selected checkpoint, and durable
  publication semantics.

### Required verification

```text
uv run pytest tests/temporal/test_history.py tests/test_min_block_fee.py \
  tests/test_modeling.py tests/cli/test_study.py
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 2 — Experiment manifests, selection, and packed launch

### Files

- `src/fable/experiments.py`
- `experiments/bundle.py`
- `experiments/launch.py`
- `experiments/feature_ablation.py`
- `experiments/c_study.py`
- `experiments/hpo.py`
- `experiments/k_study.py`
- `experiments/held_out.py`
- `src/fable/execution.py` only if required by the declared packing contract
- directly affected experiment, launch, execution, and CLI tests
- affected experiment/execution sections of `docs/FABLE.md`

### Implementation

1. Replace `ExperimentEntry`'s optional `artifact_id`, `study_id`, and `evaluation_id` fields with:

   ```python
   cell: NonEmptyString
   record_id: UUID4
   ```

   `ExperimentKind` and the manifest path define what the record ID names. Delete the validator,
   three `require_*` methods, optional serialization, and impossible mixed-reference states.
   Update all five stage scripts in one clean break. Existing local manifests are not supported.
2. Keep `tasks_per_job` with accepted values two or three. Remove the requirement that all pending
   rows divide evenly.
3. Partition pending rows into the fewest allocations at or below `tasks_per_job`, avoiding a
   singleton tail whenever at least two tasks remain. Required examples:

   | Pending | Capacity | Groups |
   | ---: | ---: | --- |
   | 7 | 3 | `3, 2, 2` |
   | 8 | 3 | `3, 3, 2` |
   | 5 | 3 | `3, 2` |
   | 4 | 3 | `2, 2` |
   | 3 | 3 | `3` |
   | 1 | 3 | `1` |
   | 7 | 2 | `2, 2, 2, 1` |

   Preserve row order. `submit_candidates`/`submit_workflows` still request one GPU per process and
   accept one to three processes as ADR 0007 specifies.
4. Keep the append-only `jobs.tsv`, flush and `fsync` after every successful allocation, printed
   job IDs, and restart skipping. Trust this program-authored ledger on reload: read submitted row
   indices and delete exact header/job/slot/cell/allocation-shape revalidation. A malformed file may
   fail naturally.
5. Make HPO the sole owner of context selection. `c_study.py` publishes the completed Study
   references only. Delete its duplicate winner computation/reporting. `hpo.py` loads those Studies,
   selects the context by the declared validation objective, and reports/uses the result.
6. Delete stage-local Study trial-count checks already guaranteed by `Study` and `TuneRequest`.
   Keep experimental cell construction, ordering, feature contracts, protocol constants, UUID
   uniqueness, stage completion gates, capacity proofs, and replay protection.
7. Consolidate Slurm script tests:
   - one exact golden allocation script;
   - packed tests assert only task/GPU scaling, payload order, and slot-specific logs;
   - CLI tests assert typed envelopes and parsed job IDs, not a second full script body.

### Test change

Tests must prove:

- each manifest kind round-trips one `record_id`;
- all stage producers and consumers use the clean schema;
- HPO alone selects context;
- packing examples above, stable row order, resume skipping, per-allocation ledger persistence, and
  a failure that leaves later groups pending;
- two- and three-GPU resource scaling and one unavoidable singleton;
- exact scientific cell counts/order/windows/methods remain unchanged.

Delete optional-ID matrices, `require_*` error tests, divisibility-error tests, ledger corruption
matrices, duplicate Study-cardinality tests, duplicate context-winner tests, and duplicate Slurm
goldens.

### Expected outcome

- One manifest reference field and no invalid entry state.
- One context-selection owner.
- Constant scheduler throughput retained: seven capacity-three fits occupy three jobs as
  `3 + 2 + 2`.
- Restartable submission remains, with substantially less validation of a self-authored ledger.
- ADR 0007's one-process-per-GPU and one-to-three-process allocation contract remains intact.

### Required verification

```text
uv run pytest tests/test_experiments.py tests/experiments tests/test_execution.py tests/cli
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 3 — Evaluation and rolling reduction

### Files

- `src/fable/evaluation.py`
- `experiments/held_out.py` only where the reducer output is consumed
- `tests/evaluation/test_resolution.py`
- `tests/evaluation/test_rolling.py`
- `tests/evaluation/test_evaluate.py` only if fixtures share the changed schema
- affected evaluation sections of `docs/FABLE.md`

### Implementation

1. Shrink `_BASELINE_RESULT_SCHEMA` to `policy` plus:
   - `base_fee_savings`;
   - `p50_fee_inclusive_savings`;
   - `base_fee_optimality_gap`.

   Remove fixed-policy action arrays and `_classification_metrics` calls from
   `reduce_baselines`. Keep learned evaluation's Accuracy, Macro-F1, MAE, MSE, and economic
   metrics unchanged.
2. Remove `reduce_rolling`'s exact nine-cell/nonempty roster guard. The held-out stage is the sole
   roster builder and already supplies the declared architecture-chain cells.
3. Remove `_ROLLING_HORIZONS` and the exact horizon-key guard. Access the required evaluations
   explicitly as `evaluation_ids[5]`, `[4]`, `[3]`, and `[2]`; missing inputs fail naturally.
4. Express the manuscript recurrence directly:
   - load `K=5` at initial origins;
   - increment current origins where action is `4`;
   - load `K=4`, then increment where action is `3`;
   - load `K=3`, then increment where action is `2`;
   - load authoritative `K=2` and use its selected fee;
   - keep `K=5` immediate, one-shot, and hindsight-minimum values.
5. In rolling validation, check only `predicted_action_k`; `minimum_action_k` is unused by rolling
   reduction and is already publisher-owned observation data.
6. Keep consecutive origin coverage, dynamic addressed-origin bounds, predicted-action bounds,
   final metric finiteness, exact observation schema/null checks, evaluation UUID/request identity,
   and exact ordered testing-window coverage.

### Manuscript acceptance oracle

The implementation must remain equivalent to:

```text
H5 = h
H4 = H5 + [prediction(K=5, H5) == 4]
H3 = H4 + [prediction(K=4, H4) == 3]
H2 = H3 + [prediction(K=3, H3) == 2]
selected = H2 + 1 + prediction(K=2, H2)
```

Every horizon runs once. Nonterminal actions do not advance the head. Every smaller horizon
supersedes the previous candidate. No `K=1`, confidence threshold, agreement vote, or moving
deadline is introduced.

### Test change

Keep compact tests for:

- the four-horizon dynamic-origin recurrence, including all-terminal and all-nonterminal paths;
- `K=2` authority and the fixed five-block deadline;
- paired one-shot versus rolling economic metrics;
- missing dynamic origin, nonconsecutive observations, and invalid predicted action;
- baseline policy ordering and exact three-metric values;
- learned evaluation's seven metrics.

Delete exact-nine-roster tests, exact-horizon-map tests, invalid rolling `minimum_action_k` tests,
nine repeated cell fixtures where one cell proves the reducer, and baseline classification
expectations.

### Expected outcome

- Baseline results describe only economics.
- Rolling code reads like the thesis recurrence.
- Workflow-owned roster structure is not revalidated inside the reducer.
- No change to selected rolling blocks or reported rolling economics for valid held-out inputs.

### Required verification

```text
uv run pytest tests/evaluation
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
```

## Slice 4 — App inference and model lifecycle

### Files

- `app/App.tsx`
- `app/src/inference.ts`
- `app/src/model.ts`
- `app/src/abort.ts` (delete when no longer imported)
- `app/src/history.ts`
- `app/src/domain.ts`
- `app/src/screens/InferenceScreen.tsx`
- directly affected app tests
- app TypeScript/config files only for dependencies made genuinely unused
- affected mobile sections of `docs/FABLE.md`

### Implementation

1. Remove `InferenceEngine.prepare` and `ModelRuntime.prepare`. Selection changes perform no RPC
   or model work.
2. Make `run(K)` the only inference path:
   - select the chain/horizon manifest;
   - synchronize the current chain context;
   - build features;
   - load the selected `.pte` if it is not already loaded;
   - run native forward;
   - decode and return the result.

   Preserve the native serial queue so load, forward, deletion, model replacement, and disposal
   cannot overlap.
3. Remove engine-owned selected horizon, desired model key, selection revisions, eager preparation,
   and `Promise.allSettled` preparation choreography.
4. Make `App.tsx` the only stale-result owner. Keep one selection/engine identity check around the
   user-visible Run result and history commit. Remove `engineRevision`,
   `engineRevisionSequence`, preparation effects, `ActiveEngine.revision`, and duplicated stale
   state.
5. Initial and post-selection inference state is idle. The Run button owns loading/error states.
   A first Run may include model-load and RPC latency.
6. Remove `outcomesRunning`. The existing serialized history queue owns outcome-update ordering;
   stale engines cannot commit.
7. Change horizon slider work to `onSlidingComplete` if it still emits intermediate selection
   changes.
8. Define `InferenceRun` as the inference result plus `id`, `ran_at`, and optional `outcome`. Use
   object spread in history creation/update while preserving the persisted JSON shape.
9. Type the native module at the narrow `load`/`forward`/`delete` contract. Trust typed native
   references and delete generic object/null/byte-buffer guards. Keep checks that prevent native
   misuse or silent bad inference: exactly two outputs, float tensors, horizon-sized logits,
   scalar minimum output, copied storage before deletion, and finite decoded values.
10. Delete `abort.ts` if no remaining caller needs a named abort error. Disposed or replaced work
    may finish, but App identity prevents it from publishing UI/history state.
11. Remove app config entries or test dependencies only when this slice makes them unused. Do not
    remove Expo peer/runtime dependencies merely because app source does not import them directly.

### Test change

Keep:

- one observable App test proving a result from an old chain/horizon cannot publish;
- one native serialization/disposal test;
- direct first-Run model load and inference;
- model replacement and copied-output lifetime;
- happy history persistence/outcome resolution;
- core prediction decoding and numerical conversion.

Delete duplicated stale-selection/preparation suites across `App`, inference, and model tests;
prepare-only tests; desired-key races; generic native object rejection matrices; and
`react-test-renderer` only if no remaining App test needs it.

### Expected outcome

- Selection is cheap and synchronous.
- One Run path replaces prepare-plus-run choreography.
- App owns stale UI state; model runtime owns native lifetime; history queue owns persistence order.
- Fewer lifecycle concepts and tests, with accepted first-Run latency.

### Required verification

```text
cd app
npm test
npm run typecheck
npx expo-doctor
```

## Slice 5 — App RPC and feature input path

### Files

- `app/src/rpc.ts`
- `app/src/features.ts`
- `app/src/domain.ts`
- `app/src/inference.ts` only for the simplified session interface
- `app/App.tsx` only for polling integration
- `app/test/rpc.test.ts`
- `app/test/features.test.ts`
- affected inference/App tests
- affected mobile sections of `docs/FABLE.md`

### Implementation

1. Make `ChainSession.sync()` perform one fresh exact-range read for the selected Run. Fetch the
   `C` context blocks plus the one predecessor needed by block-interval features, then fetch exact
   P50/P90 fee history when those features are present.
2. Delete mutable block cache state, serialized synchronization, append/trim logic, same-height
   hash probes, regressed-head recovery, and reorg refetch branches. Keep one parent-link continuity
   check across the fetched range.
3. Configure Viem HTTP transport with batching, a 10-second timeout, and zero retries. Delete
   `fetchWithTimeout`, session-signal transport wrapping, custom abort-controller plumbing, and
   related timeout/abort helpers.
4. Use Viem `watchBlocks` for the visible head/outcome polling lifecycle. Session disposal calls
   the returned unwatch function. Old in-flight reads may finish within the transport timeout;
   Slice 4's App identity gate prevents stale publication.
5. Remove one-time chain-ID verification when the production client uses the selected Viem chain's
   own default transport. Keep it only if a real supported caller can inject an arbitrary endpoint.
   Test-only dependency injection is not enough reason.
6. Trust Viem response types. Delete requested-number equality, hash-format, `requireBigInt`,
   reward-row-shape, predecessor-existence, negative-interval, gas-range, and gas-target guards
   already guaranteed by Viem, canonical manifests, and the exporter. Keep:
   - required EIP-1559 base fee when the type permits `null`;
   - parent continuity;
   - exact `feeHistory.oldestBlock` alignment;
   - final Float32 feature finiteness;
   - exact integer EIP-1559 forming-fee arithmetic.
7. Move `BlockRow` to `domain.ts` or the feature owner so `features.ts` does not import a domain
   value type from RPC infrastructure while RPC imports feature constants/types.
8. Replace the one-property `CHAIN_DETAILS` objects with direct labels if that remains a net
   deletion after the RPC rewrite.

### Test change

Keep:

- exact block range and batching behavior;
- parent-link failure;
- exact fee-history start alignment;
- Viem watcher forwarding and unwatch on disposal;
- one transport timeout configuration assertion at the client seam, not custom timer behavior;
- feature oracle parity, feature order/shape, EIP-1559 exact forming fee, and final finite Float32
  output;
- direct outcome read.

Delete cache hit/append/trim/reorg/regressed-head suites, manual polling timer tests, custom
abort/timeout tests, chain-verification tests if verification is deleted, and trusted response
shape matrices.

### Expected outcome

- RPC code is stateless application logic over Viem.
- Every Run pays for a fresh exact range, accepted for a manual demonstrator.
- Parent continuity and fee-history alignment still prevent silently mixing chain data.
- No custom network lifecycle framework remains.

### Required verification

```text
cd app
npm test
npm run typecheck
npx expo-doctor
```

## Slice 6 — Mobile exporter

### Files

- `tools/mobile-export/export.py`
- `tools/mobile-export/test_export.py`
- exporter-specific documentation

### Implementation

1. Load only `CorpusRequest` to establish each artifact's chain ID. Do not load/cache full Parquet
   corpora in the exporter.
2. Remove the duplicate roster artifact-ID uniqueness check and the exporter-local
   Ethereum-feature rule. Training associations and the chain/horizon/feature-contract checks own
   those facts.
3. Keep clean chain/horizon and shared-feature-contract checks because they prevent publishing a
   mislabeled twelve-model bundle.
4. After XNNPACK partitioning and before publishing the `.pte`, inspect the generated ExecuTorch
   program's execution-plan delegates and require at least one delegate whose ID is
   `XnnpackBackend`. Do not require every operation to delegate.
5. Keep two-input eager-versus-ExecuTorch host parity, tensor output identity, finiteness, selected
   action, decoded-fee tolerance, exact twelve-cell roster, and atomic no-overwrite directory
   publication.
6. Keep the isolated exporter environment. Core Torch and ExecuTorch's Torch version remain
   intentionally different.

### Test change

Add one focused negative test proving a portable-only program cannot pass as XNNPACK output. Keep
one successful export/parity path and atomic publication tests. Delete tests for the removed
duplicate artifact-ID and Ethereum-feature validations.

Native iOS acceptance remains outside this slice until real `MOBILE.yaml`, artifacts, and device
assets exist. Host parity is not represented as device acceptance.

### Expected outcome

- Export does not read large block corpora.
- Redundant association checks disappear.
- The one missing boundary check is added: a bundle advertised for the app must contain actual
  XNNPACK delegation.

### Required verification

```text
cd tools/mobile-export
uv run pytest
cd ../..
uv run ruff check .
uv run pyright
uv run vulture
```

## Slice 7 — Documentation truth pass

### Files

- `README.md`
- `docs/FABLE.md`
- `docs/adr/README.md`
- `docs/CONTEXT.md` only if terminology no longer matches code
- comments/docstrings made stale by Slices 1–6

### Implementation

1. Re-read the final implementation. Describe the surviving owners and paths, not the cleanup
   history.
2. Verify the historical-data description matches the surviving lazy fixed-context
   `HistoricalDataset` and its derivable origin/block arithmetic.
3. Document the single `record_id` experiment manifest schema and kind-owned interpretation.
4. Document baseline rows as three economic metrics and learned rows as seven metrics.
5. Document the direct app Run path, stateless exact-range RPC reads, Viem-owned network behavior,
   and accepted first-Run latency.
6. Document the XNNPACK delegate assertion and host-parity boundary without claiming unperformed
   device acceptance.
7. Make README command wording match behavior: submission commands submit; they do not run the
   remote candidate locally.
8. Remove nonexistent ADR 0001–0005 rows from the ADR index. Keep active ADR 0006 and 0007.
9. Replace unexplained/stale issue shorthand such as `A01` with a direct issue link or remove it.
10. Describe the dependency diagram as high-level rather than generated if no generator exists.
11. Delete stale resume wording only in Slice 8 after issue #147's gate. Until then, resume remains
    documented as current behavior.

### Test change

No prose snapshot tests. Verify links/paths manually and run the complete repository checks once.
Do not add documentation-generation or architecture-conformance scripts.

### Expected outcome

- Documentation matches final code and manuscript terminology.
- No retired ADR inventory, phantom generator, or stale module/interface description.
- No maintenance machinery is added for documentation.

### Required verification

```text
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
cd app
npm test
npm run typecheck
npx expo-doctor
cd ../tools/mobile-export
uv run pytest
```

## Slice 8 — Compact-CUDA parity

This slice runs only after Slices 1–7 are green on `main`. It reconciles
`codex/compact-cuda-execution` with finished `main`; it does not bring compact-CUDA behavior into
`main`.

### Branch operation

1. Record the pre-merge compact branch head and finished `main` head.
2. In the saved checkout, switch from clean `main` to `codex/compact-cuda-execution`.
3. Merge finished `main` into the compact branch. Do not rebase or rewrite the existing compact
   commit. Resolve every conflict so common code equals `main`; preserve only the explicit
   device-resident historical batching delta.
4. Commit the reconciliation on the compact branch.
5. After review, return the saved checkout to clean `main`.

No worktree is used.

### Allowed branch difference

`git diff main...codex/compact-cuda-execution` may contain only:

- `src/fable/_runtime.py`: removal of CPU DataLoader worker/pin/prefetch machinery that the compact
  path does not use;
- `src/fable/temporal.py`: shared backing moved once to the target device, index-returning dataset,
  and device-side batched collation/gather;
- `src/fable/modeling.py`: move prepared fit history to the Lightning root device and obtain
  training/validation loaders from the prepared datasets;
- `src/fable/evaluation.py`: move the evaluation dataset to CUDA and obtain its loader there;
- `tests/temporal/test_history.py`, `tests/test_modeling.py`, and
  `tests/evaluation/test_evaluate.py`: direct tests of those branch-only semantics;
- `docs/FABLE.md`: a compact-branch-specific description of device-resident batching, only if the
  code difference needs documentation.

Every other source, test, script, app, exporter, configuration, and documentation file must be
identical to `main`. Within the allowed files, all non-DataLoader behavior from Slices 1–7 must
also match `main`: loss ownership, deterministic/clipping ownership, tuning ownership, evaluation
semantics, guards, durable formats, and public interfaces.

### Review gate

The reviewer performs both:

- the normal Standards/Spec review of
  `git diff <pre-merge-compact-head>...<reconciled-compact-head>`;
- a line-by-line parity audit of
  `git diff main...<reconciled-compact-head>` against the allowed difference above.

`GREEN LIGHT` requires zero unrelated differences, not merely an allowed filename set.

### Required verification

Run the complete Python checks on the compact branch:

```text
uv run ruff check .
uv run pyright
uv run vulture
uv run pytest
git diff --check
```

Then return to `main`, verify a clean checkout, and record both heads and the exact remaining diff
in this ledger.

### Expected outcome

The compact branch equals finished `main` except for one isolated, reviewable implementation
choice: CPU-backed lazy loading on `main` versus device-resident historical batching on the compact
branch.

## Slice 9 — Conditional checkpoint-resume removal

This slice is not authorized to start merely because Slices 1–7 pass. It is fully specified in
[issue #147](https://github.com/edoski/fable/issues/147).

Before creating its implementer task, verify:

- every required feature-ablation, context, HPO, horizon, and authorized retry fit is complete;
- accepted Studies and artifacts are canonical;
- no queued/running job or live university checkout depends on `last.ckpt`;
- exact obsolete scratch/checkpoint paths are known.

Then remove the full-state `last` checkpoint callback, automatic `ckpt_path` resume, resumable
scratch behavior, resume-only tests, and resume documentation in one clean break. Keep Method
seeds, early stopping, weights-only best checkpoints, selected-epoch semantics, accepted outputs,
and atomic publication. Exact scratch deletion is a separate explicitly verified destructive step.

The same implementer/reviewer protocol applies directly on `main`. If this slice occurs after
compact-CUDA parity, repeat the parity procedure for its small resume deletion or retire the
temporary compact branch first under explicit user direction; do not let the branches drift
silently.

## Final acceptance

The cleanup is complete only when:

- every active slice has a committed implementer head and zero-actionable-finding Standards/Spec
  review;
- full Python, app, and exporter checks pass on `main`;
- `uv run vulture` findings have been manually classified;
- `git diff --check` passes;
- durable object and manuscript contracts listed above remain intact;
- the plan ledger contains final SHAs and review outcomes;
- Slice 8 leaves no non-DataLoader drift between `main` and the compact branch;
- Slice 9 is either completed after its gate or remains explicitly blocked by issue #147.

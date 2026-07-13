## Approval candidate: clean pre-ladder evidence implementation

This specification consolidates the closed contracts and the Issue 50 readiness audit. It is the final approval candidate for Issue 64. It incorporates Edo's separately approved TF32 amendment; approval of that amendment is not approval of this complete contract. Complete-contract approval authorizes ticket-scoped research publication, ticket publication and graph wiring, Issue 64 Resolution/closure, and the map update only. It does not authorize implementation, acquisition, data mutation, training, evaluation, the TF32 probe, remote jobs, archive work, or cutover.

## Problem Statement

Issue 49 has frozen the scientific ladder, but current `main` cannot execute it. The current workflow/config stack, chain-qualified short IDs, SQLite catalogs and artifact state, temporal compiler/policy registries, variable-window dataset path, batch-dependent loss reducers, replay evaluators, generic benchmark scheduler, snapshot-based remote execution, and mutable result collection all violate approved contracts. The required Ethereum and Polygon role-covering corpora also do not exist.

Issue 50 therefore needs one clean, reviewed, integrated pre-ladder executor before any acquisition or outcome-bearing work. The executor must construct and later run only lists 1–6: 12 Train requests and 12 post-fit validation Evaluate requests. It must not preserve legacy execution, create compatibility paths, anticipate HPO/context/final-K/testing machinery, or depend on downstream decisions reached only after Issue 50.

## Solution

Build one clean direct path from exact requests and immutable roots through causal preparation, LSTM fitting, exhaustive post-fit validation, direct Slurm execution, and six staged request-list constructors. Use a private clean-break integration branch so reviewed slices can replace cross-cutting owners without landing parallel old/new production paths. A dedicated pre-ladder integration ticket merges only reviewed slices and proves the complete path. One separate bounded synthetic TF32 evidence ticket then gates Issue 50.

The final public seams are exact typed requests, four typed root addresses/loaders, direct corpus acquisition/finalization, direct temporal and feature functions, one lazy `HistoricalDataset`, one LSTM task, one TF32-enabled-FP32 Lightning pre-ladder fit owner, one exhaustive evaluator, one per-request plan/submission owner, and six named ladder constructors. Everything else in the old path is deleted, not wrapped.

## User Stories

1. As the thesis owner, I want Issue 50 to stay blocked until the clean executor is reviewed, integrated, and passes its bounded TF32 gate, so no outcome is produced by stale or unsupported semantics.
2. As an operator, I want every Train or Evaluate identity minted once and persisted before work, so retries cannot create a different object.
3. As an operator, I want one exact request to be the whole plan, so there is no benchmark plan or duplicated workflow snapshot to reconcile.
4. As a corpus producer, I want one explicit inclusive `CorpusDefinition`, so acquisition never infers a range from features, timestamps, or a legacy recipe.
5. As a corpus consumer, I want a bare full-SHA-256 corpus ID and strict direct loader, so content and definition determine identity without a catalog.
6. As a corpus producer, I want private Parquet-prefix resume and provider-owned retry, so interruption does not create another lifecycle or nested retry owner.
7. As a scientist, I want canonical rows rejected rather than repaired after sealing, so sorting, deduplication, clipping, filling, and truncation cannot alter evidence.
8. As a scientist, I want exact closed-parent contexts and complete future outcomes, so no training, validation, or testing origin leaks across a role boundary.
9. As a scientist, I want feature and target state fitted only from their approved training populations, so validation never alters preprocessing.
10. As a model author, I want one fixed-shape lazy dataset with ordinary DataLoaders, so masks, padding, custom samplers, and duplicated context tensors disappear.
11. As a model author, I want the retained two-head LSTM and exact loss formulas, so the auxiliary head remains interpretable and loss is batch-partition invariant.
12. As a training operator, I want one native-TF32 L40 policy, deterministic seed 2026, batch 64, 36/8 stopping, strict-lower best selection, and nonfinite failure enforced by one host, so every neutral artifact follows the same protocol.
13. As a training operator, I want native completed-validation-boundary continuation without a project retry policy, so interruption recovery does not masquerade as exact continuation.
14. As an artifact consumer, I want a strict manifest, exact payload inventory, and portable CPU-FP32 best weights, so evaluation does not depend on a training checkout or private path.
15. As a reviewer, I want actual code, lock, framework, device, and Slurm identity recorded without secrets or paths, so evidence provenance is sufficient and narrow.
16. As an evaluator, I want one exhaustive traversal of the declared validation range, so every eligible origin contributes exactly once to predictive and economic totals.
17. As a scientist, I want raw integer `B/R/O/S/G/Q` accounting and additive totals, so gate results remain exact and recomposable.
18. As a scientist, I want union-active macro-F1, earliest-label accuracy, exact loss totals, and target-explicit regression diagnostics, so the frozen predictive contract is implemented without old aliases.
19. As a ladder operator, I want each next list constructible only from complete validated prior records, so a missing or invalid cell cannot freeze a winner.
20. As a ladder operator, I want exact control reuse checked from complete artifact identity, so reuse cannot be inferred from labels or approximate configuration equality.
21. As a thesis owner, I want each rung to freeze one global all-chain winner, so no per-chain hybrid or pooled-chain decision appears.
22. As a remote operator, I want exact clean-revision checks and machine-readable Slurm state, so submission never repairs or guesses cluster state.
23. As a remote operator, I want one immutable attempt history with marker-based reconciliation, so ambiguous acknowledgement is never blindly retried.
24. As a maintainer, I want the generic benchmark, catalog, compiler, prediction/evaluator registry, and compatibility machinery deleted, so the final code reflects only approved current variation.
25. As a reviewer, I want each implementation slice independently reviewed before integration, so architecture, spec, and deletion boundaries are checked separately from merge verification.
26. As Edo, I want no data, training, evaluation, job, archive, or cutover side effect in planning or implementation review, so scientific execution remains separately authorized.
27. As the thesis owner, I want the TF32 route limited to the pre-ladder executor, so Issue 26 retains final host and precision ownership without a second active host or compatibility layer.

## Implementation Decisions

### 1. Exact request and definition schema

One schema module owns frozen, extra-forbidden Pydantic values and one module-level `WORKFLOW_REQUEST_ADAPTER` over `TrainRequest | TuneRequest | EvaluateRequest`, discriminated by `workflow`. Hydration never mints. Fresh constructors mint canonical UUIDv4 destination IDs once.

`OriginWindow` is exactly `{role, first_parent_block, last_parent_block}` with inclusive bounds. `TrainRequest` is exactly `{workflow, artifact_id, source}`. Baseline source carries `{kind=baseline, corpus_id, training_definition}`; selected-study source carries `{kind=selected_study, corpus_id, study_id}`. `EvaluateRequest` is exactly `{workflow, evaluation_id, artifact_id, corpus_id, window}`. Tune remains in the approved request algebra, but no Tune execution or HPO lifecycle is implemented in this tranche.

`TrainingDefinition` stores the exact training/validation windows, `C`, `K`, ordered feature names, class-loss choice, LSTM definition, optimizer, seed, semantic batch, and fit/stopping facts. It has no runtime, provider, storage, target, device, precision, worker, Slurm, retry, affordability, metric-list, or evaluator fields.

Issue 49 narrows the older Issue 10 model-family union: the current schema contains one concrete LSTM definition only. Transformer and Transformer–LSTM types, builders, configs, and modes are deleted. The only current evidence-dependent variation is the exact ordered feature tuple and `unweighted | corrected_inverse_frequency` CE choice required by lists 1–6.

Authored recipe files, when needed, are whole strict documents loaded by one config-file owner. Surfaces, generic config groups, owner coercers, resolved fields, snapshots, selectors, and named implementation registries are deleted.

### 2. Immutable roots, canonical bytes, and provenance

Canonical addresses are only `corpora/<sha256>/`, `studies/<uuid>/`, `artifacts/<uuid>/`, and `evaluations/<uuid>.json` under one runtime storage root. Public reads are only the four typed path/load pairs. No catalog, scan, handle, generic root kind/reference, remap, or chain partition resolves an ID.

Strict manifests and evaluation records use sorted, two-space-indented UTF-8 JSON plus exactly one LF. The corpus identity preimage remains the approved compact canonical JSON of `definition` plus the UTF-8-path-sorted payload inventory. `request.json` uses the request adapter bytes plus one LF. Same ID and byte-identical object is no-op; same ID with any difference is conflict.

One private publication kernel owns contained hidden siblings, file and directory sync, exclusive no-replace visibility, parent sync, canonical reload, equality/no-op/conflict, and ambiguous post-visibility preservation. It is not a generic lifecycle or adapter.

`ExecutionProvenance` is the final common record used by Train and Evaluate results. It contains only:

- full clean repository commit;
- full SHA-256 of the locked dependency file;
- Python, PyTorch, Lightning, TorchMetrics, NumPy, Polars, and Pydantic versions;
- CUDA runtime and cuDNN versions;
- CUDA device name and compute capability;
- native positive Slurm job ID.

The request and record type already identify the workflow and destination. Precision and deterministic settings are one revision-fixed pre-ladder host contract, not a request/configuration axis and not repeated provenance. Device name, compute capability, CUDA/cuDNN versions, and the clean revision identify the actual producer; the separate bounded evidence record owns route deltas and performance facts. Store no target name, host alias, paths, log path, resource request, wall time, throughput, memory, money, environment dump, credentials, provider, timestamps, or copied batch script. This main-only tranche has one semantic/execution commit; dual-commit provenance remains conditional downstream work and creates no current field.

### 3. Corpus payload and direct acquisition/finalization

The finalized row schema and order are exactly signed-int64 `block_number`, `timestamp`, `base_fee_per_gas`, `gas_used`, `gas_limit`, and `tx_count`. Blocks are unique, contiguous, ordered by block number, use integer-second nondecreasing timestamps, positive base fee/gas limit, nonnegative gas used/transaction count, and `gas_used <= gas_limit`. Chain and regime live once in the manifest.

Canonical Parquet output is deterministic under the lock: consecutive files of at most 100,000 rows, zero-padded 12-digit inclusive block endpoints in filenames, one row group per file, Zstandard level 3, statistics enabled, 1 MiB data pages, native Polars writer, no partitioning. Runtime acquisition chunk size never changes this geometry.

One chain-generic acquisition module accepts `CorpusDefinition`, a provider, runtime concurrency/chunk values, a private stage, and the storage root. Provider calls are ordinary numbered reads; the provider alone owns finite retry/backoff. Acquisition validates completion sets before writing, orders output, cancels siblings on terminal error, and persists no counters or retry facts.

Private rows add only block hash, parent hash, and the constant strict-canonical `definition_sha256`. The exact validated Parquet prefix is the only resume checkpoint. Invalid stages are preserved and a new stage is required. Finalization validates the complete definition, schema, domains, order, links, finalized ancestry, and immediate anchor reread; strips stage-only fields; writes canonical payload; computes inventory and corpus ID; and publishes once. Existing hashless Parquet pays a fresh exact source read per row and otherwise reacquires. No Avalanche suffix and no priority-fee enrichment is added.

The old acquisition controller, adaptive split/batch retry, timestamp planning, source requirements, split materialization, mutable validation reports, acquisition run history, SQLite corpus state, chain-qualified ID, and repair paths are deleted.

### 4. Causal preparation, features, and representation

The temporal module exposes only `select_eligible_origins`, `earliest_minimum`, `select_outcomes`, `require_action`, `target_block`, and `broadcast_after_block` with small immutable results. Context is `h-C+1…h`; outcomes are `h+1…h+K`; `H=0`; every role/regime boundary follows the approved strict purge. No compiler, problem store, capability, execution policy, action space, mask, seconds geometry, or repair remains.

The feature owner implements the fixed formulas and stable order directly. It fits one independent strict float64 population `FeatureState` from the exact unique training support interval and emits float32 rows. The target owner computes raw minima/earliest labels first, then fits the separate Issue 58 float64 `TargetState`. Both states bind exact corpus, chain, regime, feature/range, count, dtype, and training provenance. There is no combined preparation state, epsilon, fallback scale, clipping, inverse input transform, or scikit-learn path.

One concrete lazy map-style `HistoricalDataset` returns CPU mappings with `inputs[C,F]`, scalar `label`, scalar `target`, `base_fees[K]`, and scalar `origin_block`. Training, validation, and testing datasets share immutable canonical row storage and retain no `N×C×F` or `N×K` table. Ordinary DataLoader owns batching, train-only seeded shuffle, default collation, workers/prefetch/persistence/pinning, and `drop_last=False`; host arguments remain ephemeral.

Live serving uses a separate `prepare_live` function sharing only the strict feature transform and action arithmetic. This transitive cutover is required so the deleted temporal/representation stack does not survive as a second path. It performs no serving expansion.

### 5. One LSTM task, exact loss, and predictive scorer

One direct LSTM builder creates the neutral model: projection 256, hidden 256, two layers, head hidden 256, dropout 0.2, K-wide classification head, and scalar regression head. The task exposes direct head construction, loss, decode, and post-fit predictive scoring functions. There is no family/prediction registry, decoded-result ABI, callable contract bundle, mask, or generic metric framework.

Unweighted CE and corrected inverse-frequency CE use unreduced native cross entropy divided by sample count. Weighted supports must be positive for all classes and weights are `N/(K*n_k)`. Native unreduced Smooth L1 enters once. Total numerator is classification plus regression numerator; complete-map loss divides once by N. The target remains Issue 58’s exact standardized log minimum.

Post-fit predictive scoring creates fresh direct TorchMetrics state for one complete range, validates every decoded action, and publishes `PredictiveResultContext` plus `PredictiveTotals`: N, classification/regression/log-error sums, earliest-correct count, and K-length target-support, prediction-count, and true-positive vectors. Scalar losses, earliest accuracy, union-active macro-F1, log MAE, and log MSE derive once. Old metric IDs, minibatch averages, stored metric state, nullable maps, aliases, and compatibility readers are deleted.

### 6. Minimum pre-ladder training host and artifact

Use one direct Lightning 2.6.5 automatic-optimization implementation with TF32-enabled FP32 on native-TF32 NVIDIA L40 hardware. Lightning `precision="32-true"` keeps inputs, parameters, outputs, losses, gradients, and AdamW state FP32 and creates no mixed-precision autocast route; eligible internal CUDA matrix and cuDNN operations may use TF32. Published weights remain portable CPU FP32, and complete-map reducers remain float64.

The host hard-requires the configured device to be an L40 with native TF32 support. It sets FP32 matmul policy `high`, enables CUDA-matmul and cuDNN TF32, and otherwise enforces seed 2026 before every stochastic construction, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, cuDNN deterministic/no benchmark, precision `32-true`, AdamW, `set_to_none=True`, clip norm 1.0, batch 64, accumulation 1, full tail batch, validation every epoch, max 36, patience 8, semantic min-delta zero, strict-lower earliest best, and failure on any required nonfinite loss or gradient norm. Unsupported hardware, a failed route gate, or any mismatch fails loudly and keeps Issue 50 blocked. There is no configurable precision axis, automatic fallback, alternate active host, or permanent selector.

This is the ordinary pre-ladder executor, not Issue 26's final host or precision selection. Its workflow seam is `train(request, storage_root, runtime_args) -> ArtifactManifest`; Issue 26 retains ownership of the later internal `fit(...) -> FitResult` host decision and may replace this single implementation behind the direct workflow seam. It may not retain both implementations or add an adapter, shim, compatibility path, or selector.

Lightning owns automatic transfer, backward, clipping, and stepping. One native best checkpoint callback writes only best weights; one separate native full last checkpoint remains private for completed-validation-boundary continuation. Partial-epoch work is discarded. The published artifact contains only the portable CPU-FP32 weights-only best checkpoint, strict manifest, and exact payload inventory—no optimizer, RNG, loader, history, last checkpoint, telemetry, or resume state.

The artifact manifest stores the exact TrainRequest once; corpus/chain/regime facts; feature and target states; class-loss state; LSTM/input/head facts; completed epochs; unique origins, optimization examples, minibatches, updates; earliest best epoch; stopping epoch/reason; best complete-validation additive loss totals; payload inventory; and `ExecutionProvenance`. It stores no duplicated recipe, evaluator, capability, registry ID, action width, runtime target/path, affordability fact, or generic metric map.

### 7. One exhaustive post-fit evaluator and evaluation record

`evaluate(request, storage_root, runtime_args) -> EvaluationRecord` strictly loads the request’s artifact and corpus, validates same-chain and exact artifact semantics, selects every eligible origin in the declared validation/testing `OriginWindow`, runs model inference once, and computes predictive and economic sections. Missing inference or any invalid/nonfinite/provenance-mismatched cell fails the whole evaluation.

Economic totals retain exact integer N, sums of B/R/O/S/G/Q, harmful-action count, selected-action counts, extra-wait-block sum, selected-trigger-wait-seconds sum, and the approved three mean-origin fraction source sums/counts. They also retain candidate/eligible counts, first/last eligible origin, and structural exclusion counts. Derived ratios are serialization/report output, not duplicated mutable facts. `S+Q=G` is checked per origin and in totals.

The immutable evaluation record stores the exact EvaluateRequest once, result coverage facts, predictive totals, economic totals, and `ExecutionProvenance`. Parent semantics derive from strict artifact/corpus loads and are not echoed. There is no evaluator ID/config, delay, replay run, sample seed, Poisson state, metric dictionary, artifact mutation, or nested evaluation under an artifact.

### 8. Direct plan, remote execution, and typed transfer

One request per plan uses the approved `request.json`, immutable intent/job facts, kernel advisory lock, marker/UID reconciliation, explicit restart, and no force override. `create_plan`, `submit_plan`, and `restart_plan` are the only plan operations. No list plan, status, target, revision, benchmark coordinate, retry count, database, or mutable pointer is durable.

Remote execution uses concrete `load_target`, `revision`, `submit`, and `follow` functions. It sends the exact request JSON on stdin, dispatches by workflow, verifies the full clean commit before submit and at job start, uses direct workflow resources, parses `sbatch --parsable`, follows `squeue --json` then `sacct -P`, and never repairs deployment. Train/Evaluate runtime provenance reads native `SLURM_JOB_ID`; execution adds no envelope.

Typed transfer directions needed now are push finalized corpus and pull immutable artifact/evaluation. Each uses its direct loader, hidden stage, rsync, destination validation, no-replace publication, and reload. There is no generic transfer kind/result, replace option, remote catalog, or path envelope.

### 9. Six staged ladder constructors and gate

One `ladder` module owns six named constructors only:

1. capacity/activity Train: six, chain-major with lean/complex inner;
2. capacity/activity validation Evaluate: six in matching order;
3. UTC Train additions: three, only after the capacity winner is computed;
4. UTC validation Evaluate additions: three, reusing the selected prior controls;
5. CE Train additions: three, only after the UTC winner is computed;
6. CE validation Evaluate additions: three, reusing exact unweighted controls.

Each constructor produces complete ordinary requests and immediately hands each request to the one-request plan owner. It persists no list, stage, label, coordinate, dependency edge, callback, watcher, scheduler, or resume state.

One pure gate function loads the required six exact validation records, proves chain set/order, identical paired origins and non-candidate identity, and valid complete evidence. Lean/default wins globally only when all three chains meet captured opportunity within 0.05 and harmful-action rate does not increase; otherwise complex wins. Any missing, failed, nonfinite, provenance-invalid, undefined-`sum(G)`, or mismatched cell yields no winner. The next constructor consumes the computed winner and exact control IDs. No separate winner database or per-chain hybrid exists.

The entire generic benchmark package, benchmark command/config group, active benchmark YAMLs, run codecs, collection snapshot, SQLite index, search/scan/export, and benchmark tests are deleted. Historical bytes and unrelated research scripts remain untouched.

### 10. Clean-break integration and graph

Implementation and review happen on a private clean-break integration branch. No implementation slice lands on `main`, runs scientific work, or exposes a parallel production mode. Every slice has one separate code-review task. A review either approves the exact slice or sends it back; it performs no implementation.

A new dedicated pre-ladder integration ticket merges only approved slices, resolves mechanical conflicts without changing behavior, runs the complete verification suite, proves zero deleted-path imports/configs, and reports the integrated commit. Any substantive correction returns to a new or reopened implementation plus review task. This integration ticket blocks the separate precision-evidence ticket and the later full integration Issue 65. Every implementation ticket is a direct child of the map and directly blocks Issue 65 as required.

After the reviewed host is integrated, one separate bounded precision-evidence execution ticket runs only Issue 62's frozen synthetic full/tail LSTM gate on the actual native-TF32 L40: native capability and representative-operation support; tensor, parameter, output, loss, gradient, optimizer-state, and checkpoint dtypes; finite loss and gradient norm; one actual optimizer update; deltas from cloned strict-FP32 reference weights; synchronized cold/steady timing and peak allocated/reserved memory; and deterministic repeats on the same locked host/runtime. It uses no thesis data and inspects no predictive or economic outcome. It is the immediate native blocker of Issue 50. Unsupported hardware, an operation/dtype/finiteness/update invariant failure, or repeatability failure leaves the ticket and Issue 50 open and returns the decision to Edo; it never falls back to strict FP32 or another host. Delta, timing, and memory values are reported as frozen evidence and create no post-hoc threshold.

All twenty-two resulting tickets are direct children of the map. The ten implementation tickets and the pre-ladder integration ticket directly block Issue 65. The precision-evidence ticket blocks Issue 50 only; Issue 26 retains the later final host/precision decision.

Dependency order is:

```text
benchmark deletion
  -> request algebra
  -> immutable root/publication kernel
       -> corpus acquisition/finalization
       -> causal preparation/representation
            -> LSTM task/loss/scorer
                 -> Lightning training/artifact
                      -> exhaustive evaluation
  -> direct plan/remote execution/typed transfer
       -> six ladder constructors/global gate

each implementation -> its independent review
all reviews -> pre-ladder integration -> TF32 precision evidence -> Issue 50
every implementation + pre-ladder integration -> Issue 65
```

The ticket breakdown after approval contains ten implementation tickets, ten corresponding review tickets, one pre-ladder integration ticket, and one bounded precision-evidence execution ticket: twenty-two tickets total. There is no umbrella implementation ticket, and the evidence execution is not combined with implementation, review, integration, acquisition, or Issue 50.

## Testing Decisions

Tests cross the highest surviving owner seam and replace deleted tests. Keep:

- one request-adapter/config-file behavior matrix;
- one canonical identity/JSON/publication golden group;
- one fake-provider acquisition/resume/finality/publication fixture plus invalid-stage table and existing-Parquet reacquire case;
- one geometry/outcome fixture and the two approved preprocessing/dataset/offline-live fixtures;
- one exact task-loss partition test and one complete predictive-scoring absent-class test;
- one synthetic full/tail Lightning fit group covering exact `32-true`/TF32 backend configuration, fail-loud capability handling, FP32 state and float64 reducers, deterministic construction, strict best/36–8 stopping, nonfinite failure, private boundary continuation, and portable best-weight reload;
- one hand-computable exhaustive evaluation fixture proving every origin once and `S+Q=G`;
- one plan/submission/reconciliation/target/follow/typed-transfer group with no real Slurm call;
- one ladder count/order/reuse/global-gate group;
- full `pytest`, Ruff, Pyright, and `uv run vulture`, with every Vulture finding manually checked for dynamic/framework/CLI/config references before deletion.

Delete transition, old/new parity, migration, registry/codec round-trip, architecture-deletion, legacy-shape, worker-heuristic, exact-resume, Poisson/replay, benchmark-index, catalog, selector, mask/padding, and compatibility tests. Tests create only synthetic in-memory or temporary-directory facts. They perform no RPC, production storage mutation, training evidence, evaluation evidence, or scheduler job. Permanent tests do not claim real L40 capability, timing, memory, deltas, or repeatability; the separate one-time precision-evidence ticket owns those facts.

## Out of Scope

- Actual Ethereum/Polygon acquisition or Avalanche import.
- Role-window derivation or freezing against real new corpora.
- Any Issue 50 training/evaluation or ladder winner.
- Tune execution, HPO policy/search/promotion, context study, final-K training, sealed testing, final TSV, accelerator evidence beyond the bounded pre-ladder Issue 62 TF32 gate, serving expansion, mobile execution, data acquisition, archive movement, or cutover.
- Final CLI design, final dependency policy, final host/precision alternatives, dual-commit fast execution, study lifecycle, final serving records, or later evidence-dependent task choices.
- Compatibility readers, migrations, old-root conversion, dual writes, aliases, dormant registries, provisional schemas, affordability/profiling systems, automatic fallback, or resource projections.
- Mutation or cleanup of unrelated dirty work, especially `benchmarks/scripts/render_lstm_block_count_quartile_results.py` and `docs/research/issue-50/`.

## Further Notes

`CONTEXT.md` and ADRs 0001, 0002, 0004, and 0005 describe superseded catalogs, compiler/materialization, snapshots, benchmark, and Session vocabulary. Implementation must not preserve those contracts merely because documentation remains stale; Issue 39 owns final normative language. ADR 0003’s representation responsibility survives, but its adapter/persisted-representation wording does not.

Approval means this contract is complete enough to publish the code-review-sized implementation/review/integration tickets plus the separate precision-evidence execution ticket, wire native dependencies, publish ticket-scoped Issue 64 research, close Issue 64 with one Resolution, and update the map. It does not authorize implementation, the TF32 probe, or any scientific execution.

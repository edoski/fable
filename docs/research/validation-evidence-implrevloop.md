# Validation evidence and experiment refresh implementation-review ledger

Status: implementation complete; both slices green

This ledger standardizes validation evidence across model-selection stages, extends the context
study below 25 blocks, and defines the clean rerun sequence. Cost over optimum remains the sole
selection objective. Other metrics are evidence and interpretation, not additional selectors.

## Pre-run state

- Checkout: `/Users/edo/dev/python/fable`
- Branch: `main`
- Planning baseline: `8ff672f0fb841fefa51a5b52135d08aaddd424cb`
- Approved-ledger commit: `bbf418ba7efa8f9f051a4976725d3154392f082d`
- Checkout policy: direct `main`, no worktree, one writer at a time
- Protected unrelated work:
  - modified `experiments/figure_context_study.py`
  - modified `experiments/figure_feature_ablation.py`
  - modified `tools/mobile-export/uv.lock`
  - untracked `docs/experiments/`
  - untracked `docs/research/macos-inference-energy.md`
- Concurrent work gate: the inference-benchmark ledger and planned KAIROS rename must not write
  concurrently with this run.
- External state: the current 45-cell Stage 2 campaign remains live. This planning turn does not
  cancel, repack, finalize, extend, or launch jobs.
- Existing Stage 1 remains immutable until a replacement campaign has closed and passed checksum
  and loader verification.

## Approval

The user approved the complete ledger on 2026-08-03:

- canonical selected checkpoint plus validation observations;
- `last.ckpt` retained only while a fit is incomplete;
- full Stage 1 rerun and replacement of the current Stage 2 campaign;
- two implementation slices on direct `main`, without worktrees; and
- metric implementation before the separate KAIROS rename.

## Superseded campaign archive

Completed 2026-08-03 before Slice 1:

- Cancelled only running jobs 43568–43570 and pending jobs 43571–43573. Completed jobs
  43565–43567 were not altered.
- Archive: `/Users/edo/Desktop/fable-pre-validation-evidence-refresh-2026-08-03`.
- Archived the 102 local canonical Stage 1 Studies and manifest, current 45-cell Stage 2 bundle,
  every recoverable remote object for the 36 new Stage 2 Study IDs, and logs 43565–43573.
- All three gzip streams passed integrity checks; local and remote SHA-256 values matched.
- Moved the current Stage 2 authoring bundle out of repository outputs.
- Removed only the archived 36 Stage 2 remote Study targets and exact job logs. Remote Stage 1
  remains exactly 102 canonical Studies; hidden Studies and the superseded job queue are empty.
- Deleted the obsolete Stage 2 heartbeat after the queue and active bundle were removed.

## Fixed scientific decisions

### Shared metric vocabulary

Every selected validation checkpoint in Stages 1–4 must retain enough per-origin evidence to
recompute the same metrics already used by held-out evaluation:

1. accuracy;
2. macro-F1;
3. log-fee MAE;
4. log-fee MSE;
5. base-fee savings versus immediate execution;
6. P50 fee-inclusive savings versus immediate execution; and
7. base-fee optimality gap, presented as **Cost over optimum**.

Cost over optimum remains the primary selection metric for feature selection, context selection,
and HPO. The additional metrics must not create a multi-objective or post-hoc selection rule.

Store raw fees in wei/gas and convert only for presentation. Validation reports may additionally
derive mean immediate, selected, oracle-minimum, and selected-minus-minimum base fee in Gwei/gas.
These absolute values are chain-specific descriptive context, not a fair cross-chain monetary
comparison: native-token value and transaction gas use differ.

### Evidence retention

The durable evidence for each completed fit is:

- the selected best checkpoint;
- canonical validation observations for the selected checkpoint; and
- the compact result metadata already needed for selection and epoch provenance.

The full-state `last.ckpt` exists only in hidden scratch while a fit is incomplete and resumable.
Delete it after successful publication. Do not retain superseded checkpoints, candidate result
fragments, caches, logs, or operational scratch. Publication converts the selected checkpoint and
observations into one canonical Study or artifact object and then removes hidden scratch. This
preserves future re-reduction and exact selected-model inference without turning `outputs/` into an
unstructured training directory.

The validation observation schema should be shared with held-out evaluation so one reducer owns
metric definitions. No stage-specific metric implementation is allowed.

### Context study

Stage 1 uses `C=25` as its reference geometry. Stage 2 reuses those nine exact selected-feature
Studies, then evaluates the remaining contexts. Stage 3 and later stages use the context selected
for each chain by Stage 2.

The frozen context roster is:

`C = {1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, 400}`.

`C=1` is intentional. It gives the model only the latest closed block. LSTM recurrence and
Transformer attention then have no temporal sequence to compare, so it is the clean minimum-history
baseline. `C=2` is the first genuinely temporal sequence.

For each chain, average Cost over optimum equally across LSTM, Transformer, and Hybrid. Let `m(C)`
be that chain mean and `m* = min_C m(C)`. Tentatively select the smallest tested context satisfying

`m(C) <= 1.05 * m*`.

This is a 5% relative tolerance, not five percentage points. Report the complete curve and the
unconstrained best context beside the selected smallest context so the rule remains inspectable.
Tie-break toward smaller `C`.

Do not test adaptive intermediate values yet. Record as deferred: after the fixed roster completes,
inspect whether adjacent tested values bracket a scientifically useful threshold, then approve any
refinement before running it.

### Training semantics protected from change

- Same corpora, windows, feature routes, seeds, models, loss, precision, epoch limit, and patience.
- Early stopping may continue to monitor validation total loss.
- Best-checkpoint selection continues to monitor Cost over optimum.
- Held-out testing remains sealed.
- No current result is retroactively enriched or presented as if it contained missing evidence.

## Canonical output direction

Use one complete canonical object per Study rather than permanent hidden candidate scratch. A
target shape is:

```text
studies/<study-id>/
  study.json
  trials/<method-index>/
    selected.ckpt
    validation.parquet

artifacts/<artifact-id>/
  artifact.ckpt
  validation.parquet
```

Exact filenames may change during Slice 1 if a smaller direct ownership model emerges, but the
observable contract may not: one canonical selected checkpoint plus self-contained validation
observations must exist, and resumable hidden scratch must not become durable output.

This clean break conflicts with ADR 0006's current flat Study JSON and single artifact checkpoint
addresses. Slice 1 must update the ADR explicitly; it must not add compatibility readers or path
shims. Old experiments are archived before replacement.

## Slice 1 — Canonical fit evidence and shared reduction

Status: green

### Implementation and review record

- Baseline: `354a15a182cd706cbdaf01390e6cc3a9d7036b33`
- Implementation: `f6f018f2bea762a9affb41d514f4ea4256fd7942`
- Correction: `077584553fecfff4bd44e453fdb703ac902649bf`
- Implementer: `/root/slice1_validation_evidence`
- Reviewer: `/root/slice1_review`, with independent Standards and Spec axes
- Initial review: rejected for a held-out precision regression, missing artifact epoch provenance,
  and an omitted macro-F1 formula.
- Correction review: `GREEN LIGHT`; zero actionable Standards or Spec findings. Held-out inference
  remains FP32, selected-fit validation uses the established family precision, and artifact
  checkpoint, observations, and compact result provenance publish atomically.
- Verification: 110 tests passed; Pyright, Vulture, changed-file Ruff/format, and diff check clean.

### Scope

- Deepen evaluation observation collection so validation and testing share one prediction/outcome
  schema and one metric reducer.
- After fitting, run one deterministic inference pass from the selected checkpoint over the exact
  validation window.
- Publish selected checkpoint, validation observations, result metadata, and request association
  atomically as the completed Study trial or trained artifact.
- Make Stage 1, Stage 2, Stage 3, and future Stage 4 fits inherit the contract through the shared
  fitting path.
- Derive the seven fixed metrics and Gwei summaries from observations. Do not duplicate aggregates
  in multiple owners unless publication needs one explicit objective-equality check.
- Update `docs/CONTEXT.md`, ADR 0006, loaders, addresses, and focused tests for the clean-break
  object shape.

### Non-goals

- No experiment grid, feature route, HPO roster, training objective, early-stopping, precision,
  inference benchmark, RPC, mobile, manuscript, plotting, or remote-job change.
- No legacy flat-Study or artifact-path reader.
- No per-epoch prediction archive or learning-curve subsystem.
- No retention of arbitrary scratch beyond selected, resumable, and observational evidence.

### Protected behavior and accepted tradeoffs

- Candidate resume remains full-state and deterministic.
- Failed candidates preserve scratch for diagnosis and resume.
- Atomic no-overwrite publication and strict request/method alignment remain.
- One extra selected-checkpoint validation pass is accepted. It avoids retaining predictions from
  every epoch and makes the reported evidence correspond exactly to the chosen checkpoint.
- Storage growth is accepted. Current checkpoints indicate roughly 3.7–11.1 MB for selected
  weights; validation Parquet will dominate. Measure one fit per chain before projecting the final
  total.

### Expected outcome

Any completed fit can be reduced later into the complete fixed validation metric set without
retraining, parsing logs, or requiring hidden scratch. Completed outputs remain canonical and
compactly grouped.

### Checks

- Exact selected-checkpoint prediction and result-objective equality.
- Validation and held-out reducer parity on identical observations.
- Accuracy, macro-F1, log-fee errors, economic metrics, and Gwei summaries.
- Atomic candidate/artifact publication, interruption, resume, and no-overwrite behavior.
- Strict Study/method and artifact/request association.
- Focused modeling, Study, artifact, evaluation, and experiment tests.
- Full pytest, Ruff, format, Pyright, Vulture with manual classification, and diff check.

### Review gate

A fresh implementer uses the `implement` skill and commits on direct `main`. A separate fresh
reviewer pins the slice baseline/head and uses `code-review` for independent Standards and Spec
axes. Rejected findings return to the same implementer; corrections return to the same reviewer
until `GREEN LIGHT`.

## Slice 2 — Stage protocol, lower contexts, and reporting

Status: green

### Implementation and review record

- Baseline: `c1d8d67c4ec18f229d38dfbb8b56d02805ab3e83`
- Implementation: `7ddaec98f5635109e3831d4fcc053a4d11a599cc`
- Corrections: `f071904e4977eee65839b81c31c20b5782a4ff67`,
  `40ffb6c915414bb80b30dd387b0581e54bcb6b03`, and
  `3a5f5fc3fe8254388ee3e3e0b242e27de2835199`
- Implementer: `/root/slice2_stage_protocol`
- Reviewer: `/root/slice2_review`, with independent Standards and Spec axes
- Review loop removed quadratic Study reduction and redundant figure-test work, then closed source
  identity gaps exposed by the fixtures.
- Final review: `GREEN LIGHT`; zero actionable Standards or Spec findings. Selection requires one
  aligned trial per cell, exact family/context labels, one corpus and normalized Experiment identity
  per chain, and one Method identity per chain-family. Only context may vary across the Stage 2
  curve.
- Verification: 114 tests passed; Pyright, Vulture, changed-file Ruff/format, and diff check clean.

### Scope

- Expand Stage 2 to the fixed 13-context roster.
- Reuse the nine exact new Stage 1 selected-feature `C=25` Studies; author the other 108 Stage 2
  fits.
- Select the smallest context within the tentative 5% relative chain-mean tolerance, with smaller
  context as the tie-break.
- Keep full curve, unconstrained best, threshold, and selected context available to reporting.
- Ensure Stage 1, Stage 2, and Stage 3 reports expose the shared metric reducer while selecting only
  by Cost over optimum.
- Keep Stage 3 at nine Studies and 81 HPO methods, now sourced from the selected context under the
  new rule.
- Update focused experiment and figure tests for exact rosters and selection.

### Non-goals

- No adaptive contexts between tested values.
- No exhaustive feature search, new features, changed feature route, HPO expansion, Stage 4 start,
  held-out evaluation, RPC benchmark, or manuscript prose.
- No hard-coded table values or separate stage-specific metric calculations.

### Expected outcome

One Stage 2 manifest contains all 117 aligned chain-family-context cells and chooses the shortest
tested context whose chain-level economic score is within 5% of the best. Stage 3 consumes that
choice directly and every candidate retains the same validation evidence.

### Checks

- Exact contexts and 117-cell Stage 2 roster.
- Exact reuse of nine `C=25` Study IDs and 108 newly authored fits.
- Three-family equal-weight means, relative tolerance, deterministic smaller-context tie-break,
  unconstrained-best reporting, and full-curve availability.
- Nine Stage 3 Studies, 81 unique methods, strict source chain/family/context alignment.
- Focused Stage 1/2/3, launch, bundle, figure, and reducer tests.
- Full pytest, Ruff, format, Pyright, Vulture with manual classification, and diff check.

### Review gate

Use a new implementer and a new reviewer. Pin the green Slice 1 head as baseline. Apply the same
implementation-review-correction loop and advance only on `GREEN LIGHT`.

## Deployment and campaign phases

These are external execution gates after both code slices are green; they are not code-review
slices.

### Gate A — Identity cutover

The planned clean-break FABLE-to-KAIROS rename is a separate task. Preferred order:

1. finish both metric slices;
2. perform and review the KAIROS rename on top;
3. build and smoke-test one immutable KAIROS CUDA image; and
4. run refreshed campaigns only from that image and the new KAIROS paths.

This avoids building an intermediate FABLE image and later migrating fresh experiment outputs.
Historical manifests and archives retain their original identity and hashes.

### Gate B — Superseded-output archive and reset

After explicit approval, cancel only the exact current Stage 2 live/pending jobs. Copy the current
Stage 1, current partial Stage 2, and any recoverable stale Stage 3 records outside the repository
into a dated checksum manifest. Verify the archive before removing exact stale local/remote targets.
Never delete corpora.

### Phase 1 — Stage 1 rerun

- Run all 102 feature-ablation fits under the new evidence contract.
- Expected compute: roughly 400–550 GPU-hours from the original campaign. With sustained 12-GPU
  occupancy and one added selected-checkpoint validation pass, budget about 36–60 wall-clock hours,
  plus queue and final-tail variance.
- Verify 102 canonical Studies, complete validation evidence, finite metrics, checksums, and one
  manifest-only experiment directory locally.

### Phase 2 — Expanded Stage 2

- Reuse the nine exact Stage 1 `C=25` Studies.
- Run 108 new fits for the other 12 contexts across three chains and three architectures.
- Close one 117-cell manifest, reduce the complete metric table, and freeze the selected context per
  chain under the approved tolerance.

### Phase 3 — Stage 3 rerun

- Author nine Studies and run 81 HPO candidates from the selected Stage 2 contexts.
- Retain complete evidence for every method, not only the winner.
- Stop at 81/81. Selection/finalization and Stage 4 remain a user gate.

Rough refreshed Stages 1–3 total: 291 new fits. Budget approximately four to six days at sustained
10–12 GPU occupancy, with queue topology and long-tail fits able to extend it. Replace this estimate
after the first wave measures the added evidence-pass overhead.

## Sibling inference benchmarks

The existing inference-benchmark ledger remains separate.

- Controlled compute latency and live end-to-end latency are sibling benchmarks.
- Compute reports mean with 95% CI plus median, p95, p99, and observed maximum.
- Live end-to-end reports RPC acquisition, feature preparation, model+decode, and total latency with
  the same distribution summaries plus success, timeout, and failure rates.
- Energy and monetary inference cost do **not** get per-call p95/p99 from one-second `powermetrics`
  samples. The independent values are paired phase estimates; report mean with 95% CI and optionally
  median/IQR. Convert energy estimates and CI endpoints linearly to EUR.
- One-shot versus rolling physical workloads remain a deferred protocol discussion.

## Deferred decisions

- Whether 5% remains the final context tolerance after reviewing the full curve. Changing the
  tolerance later requires no rerun.
- Whether fixed-grid neighbors justify a separately approved refinement run.
- Exact live RPC provider, cold/warm protocol, and one-shot-versus-rolling layout.
- Final KAIROS rename schedule and repository/remote cutover window.

## Approval gates

Both approved implementation slices received `GREEN LIGHT`. A final integrated run passed 114
tests, Pyright, Vulture, changed-file Ruff/format, and the full-range diff check. External execution
remains gated on the separate KAIROS rename completing and the resulting immutable CUDA image
passing container and GPU smoke tests.

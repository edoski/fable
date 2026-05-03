# Spice

Spice is a temporal fee-decision research pipeline for EVM chains. This context names the project concepts used when config selections become executable workflow configs.

## Language

**Surface**:
A named recipe that groups default config choices for one workflow family.
_Avoid_: preset, profile

**Workflow Selection**:
Unresolved workflow intent made of a surface plus optional overrides.
_Avoid_: request

**Root Consumer Selection**:
Explicit existing-root id intent for a workflow or action that reads an existing corpus, study, or artifact.
_Avoid_: config echo, fuzzy selector

**Storage Operator Outcome**:
Renderable result produced by storage-owned show/delete command logic for an existing persisted root.
_Avoid_: CLI storage policy, print helper result

**Producer Root Identity**:
Deterministic id for a corpus, study, or artifact a workflow will produce from semantic provenance.
_Avoid_: output path, generated selector

**Workflow Config**:
Executable typed configuration produced by resolving a workflow selection.
_Avoid_: resolved request

**Concrete Owner Config**:
The concrete local-spec config type selected by an owner id and safe for owner coercers to preserve by identity.
_Avoid_: selector base config, typed payload

**Resolved Workflow Hydration**:
Config module that turns a raw Resolved Workflow Snapshot back into a typed train, tune, or evaluate Workflow Config through owner coercers, without re-resolving a Surface.
_Avoid_: workflow config coercion, surface replay

**Resolved Workflow Snapshot**:
Durable JSON payload for a resolved train, tune, or evaluate Workflow Config, loaded through owner coercers without re-resolving surfaces.
_Avoid_: config dump, hydrated request

**Config Group**:
A named collection of YAML specs for one kind of selectable configuration.
_Avoid_: config bucket, config folder

**Config Group Loading**:
Config registry Interface that loads named Config Group entries either as canonical raw payloads for editing/display or as typed owner configs for workflow resolution.
_Avoid_: YAML helper, generic config loader

**Problem Spec**:
The typed temporal problem definition selected by a workflow.
_Avoid_: problem config, problem preset

**Feature Catalog**:
Registered feature implementation selected by `features.id`. It owns source availability, formulas, selectable outputs, prerequisites, and fingerprint sources for one feature set.
_Avoid_: feature allow-list, output slice

**Temporal Capability**:
Typed value carried by trained artifacts that bundles the problem compiler runtime metadata with the artifact action width and maximum supported delay.
_Avoid_: max slots, compiler metadata payload

**Tuning Execution**:
Modeling-owned runtime that opens a compatible study, runs Optuna trials, writes trial metadata, and returns the storage-owned study summary.
_Avoid_: tune workflow objective, study storage helper

**Benchmark**:
A named matrix of workflow selections used to compare experiment variants.
_Avoid_: experiment batch

**Benchmark Case**:
A benchmark subdivision that shares base selection values and dimensions.
_Avoid_: scenario

**Benchmark Step**:
One workflow action inside a benchmark case.
_Avoid_: task

**Benchmark Plan Entry**:
One durable executable benchmark row with run id, dependencies, selection ledger, and a Resolved Workflow Snapshot.
_Avoid_: benchmark workflow selection, expanded row

**Benchmark Dependency Ledger**:
Benchmark-owned durable scheduling facts for a Benchmark Plan Entry: matched local run ids, external Slurm dependencies, and the `artifact_from` source run when present.
_Avoid_: depends-on tuple, submission helper state

**Benchmark Root Ledger**:
Benchmark-owned durable root facts for a Benchmark Plan Entry, separating consumed root ids, produced root ids, and artifact-source dataset identity from benchmark selection coordinates.
_Avoid_: injected selection ids, materialization state dict

**Benchmark Plan Materialization**:
Benchmark module that turns Benchmark Specs, Cases, and Steps into Benchmark Plan Entries by expanding dimensions, matching dependencies, deriving dependency-produced root ids, and resolving Workflow Config snapshots.
_Avoid_: benchmark compilation helper, id patching

**Benchmark Run**:
One durable benchmark plan with local run-state files for metadata, plan, submission, and collection snapshot.
_Avoid_: ad hoc benchmark output folder

**Benchmark Plan Execution**:
Two-step benchmark operation where `plan` creates durable run state without remote access and `submit` executes exactly that persisted plan remotely.
_Avoid_: hidden replanning, submit-by-name

**Benchmark Collection Snapshot**:
Complete all-or-nothing `collection.json` for one Benchmark Run. Contains only successful expected evaluate results and is replaced only after every expected result resolves.
_Avoid_: partial ledger, skipped rows

**Evaluation Execution Provenance**:
Persisted remote evaluate-job identity attached to artifact evaluation state, including execution ref, job id, log path, workflow task, and target when available.
_Avoid_: latest artifact state, evaluator-only match

**Evaluation Config Snapshot**:
Immutable normalized JSON-ready evaluator config persisted with artifact evaluation state. It is derived from the active evaluate Workflow Config's evaluator config and excludes delay, root ids, batch size, storage, and full workflow identity.
_Avoid_: live evaluator config, evaluation workflow snapshot

**Benchmark Result Record**:
One summary-level benchmark observation from a collected evaluate result, including benchmark coordinates, submission facts, artifact/evaluation identity, and aggregate metrics.
_Avoid_: raw replay dump, CSV row

**Benchmark Result Index**:
Rebuildable SQLite projection over Benchmark Collection Snapshots used for small operator queries and named CSV exports. Run dirs remain source of truth.
_Avoid_: canonical results database

**Storage Selector**:
A typed query for one existing catalog record, preferably by exact root id: dataset, study, or artifact.
_Avoid_: workflow selector

**Root Handle**:
Resolved runtime reference to a storage root after catalog lookup or deterministic producer identity. Carries root id/name, chain, root path, state database path, root-specific identity, and storage-owned root operations needed by workflows.
_Avoid_: path bag, catalog record

**Produced Root Handle**:
Root Handle for a not-yet-existing or staged workflow output, derived from Producer Root Identity and canonical layout.
_Avoid_: workflow paths

**Root Lifecycle**:
Validation, staging, promotion, partial commit, reindex, and delete behavior for storage roots.
_Avoid_: storage sync

**Storage Transfer Transaction**:
Execution-owned push/pull behavior that resolves a catalog root, stages it, rsyncs it, promotes it with root-kind validation, and cleans up failed stages without hiding the primary failure.
_Avoid_: sync helper, rsync wrapper

**Remote Catalog Record Codec**:
Strict JSON envelope for catalog records crossing the SSH transfer seam. It carries the storage root kind plus one dataset, study, or artifact catalog record whose field shape is owned by the catalog root-kind registry.
_Avoid_: record dict, sync payload

**Corpus Assembly**:
Acquisition-to-corpus policy that plans block windows, materializes history/evaluation splits, writes corpus provenance, and publishes a corpus root.
_Avoid_: corpus builders

**Corpus Acquisition Stage**:
Corpus module that owns hidden acquisition staging roots, split sequencing, state database staging, commit cleanup, and preserve-on-failure behavior during acquisition.
_Avoid_: staging helper, acquire workflow body

**Corpus Capability Planning**:
Corpus policy that compiles feature/problem capability requirements, plans acquisition windows, counts valid temporal capability samples, and decides bounded history refills.
_Avoid_: history helper, acquisition scheduler

**Corpus Acquisition Source Requirements**:
Corpus-owned policy facts derived from feature/problem capability planning that tell concrete acquisition adapters which optional raw source fields are required for a corpus pull.
_Avoid_: workflow feature-source switch, provider policy

**Corpus Split Materialization**:
Corpus module that fulfills Split Intents through a materialization session, reusing, extending, rebuilding, and validating canonical history/evaluation block datasets. It owns internal materialization policy, staged/committed fact collection, target matching, pull execution, parquet IO, and validation. Extension reuses whole clean parquet chunks and rewrites only missing or edge ranges.
_Avoid_: parquet helper, acquisition pull

**Split Intent**:
One requested history or evaluation corpus split materialization, including target split kind, block plan, output path, and staging path.
_Avoid_: split mode flags, parquet request

**Staged Split Resume**:
Corpus Split Materialization rule that a staged split is reused only when it is clean and validates against the current Split Intent. Invalid staged data is fatal; clean staged data for another Split Intent is ignored.
_Avoid_: best-effort resume, partial cache

**Artifact Inference Context**:
Trusted inference inputs reconstructed from a trained artifact manifest, selected corpus manifest, and eval-only controls.
_Avoid_: evaluation setup, scoring setup

**Temporal Dataset Preparation Interface**:
Modeling seam that turns canonical block frames, temporal contracts, builder runtime metadata, scaler state, and split/window policy into prepared training or inference datasets.
_Avoid_: dataset plumbing, feature loading

**Action Space**:
Set of decoded actions that an execution policy can resolve for a temporal sample. Prepared once for selected samples and shared by representation inputs, prediction targets, and decode context.
_Avoid_: candidate rows, logits width

**Batch Plan**:
Executable model-batch iteration plan with sample ordering, target binding, and host/device storage-mode choice.
_Avoid_: batch source

**Training Runner**:
Fit execution module that owns runtime setup, epoch execution, objective tracking, best-state selection, and split metric evaluation.
_Avoid_: training loop helper

**Forward Runtime Plan**:
Modeling module that owns forward-only host warmup, CUDA memory measurement, final batch planning, and model forward execution.
_Avoid_: ad hoc inference probe

**Training Runtime Plan**:
Modeling module that owns the destructive gradient-bearing training probe, restores model state, and returns the reusable prediction training state plus planned runtime context.
_Avoid_: warmup side effect

**Evaluation Scoring Runtime Plan**:
Modeling plan for model-bound evaluator scoring. It carries resolved device, precision, representation runtime context, backend determinism, and seed into inference scoring.
_Avoid_: scoring batch size, inference setup

**Training Fit Policy**:
Training Runner internal policy for finite metrics, history append order, objective improvement, best-state tracking, progress payloads, and early-stop decisions.
_Avoid_: callback logic

**Decoded Result ABI**:
Typed prediction output contract consumed by evaluators after model inference.
_Avoid_: logits, prediction tensor

**Objective Runtime**:
Modeling-owned module that pairs a policy-only objective contract with the metric production used by the Training Runner.
_Avoid_: objective evaluator, objective callback

**Temporal Replay Runner**:
Evaluation-owned module that validates decoded replay inputs, asks a replay Adapter for selected temporal decision events, and delegates fee outcomes to Temporal Accounting.
_Avoid_: replay helper, evaluator base class

**Temporal Accounting**:
Evaluation-owned module that computes realized, baseline, optimum, and economic metrics for selected temporal fee decisions.
_Avoid_: fee accounting, replay accounting

**Workflow Command Selection**:
Operator-edge construction of typed workflow selections from explicit CLI values before config resolution.
_Avoid_: CLI request builder

**Benchmark Collection Resolver**:
Benchmark module that consumes a pulled artifact root plus the submitted evaluate record and selects the matching evaluation summary.
_Avoid_: benchmark artifact loader

**Execution Session**:
Target-bound SSH/SLURM session for remote commands, module execution, rsync transfer, workflow submission, job following, and remote metadata lookup.
_Avoid_: execution backend

## Relationships

- A **Workflow Selection** references exactly one **Surface** when it produces a new root.
- A **Root Consumer Selection** references existing roots by exact id and does not reproduce their original config identity.
- A **Producer Root Identity** names a produced root before the root is committed.
- A **Surface** references one or more **Config Groups**.
- A **Workflow Selection** may override values from its **Surface**.
- A **Workflow Config** is produced from exactly one **Workflow Selection**.
- A **Resolved Workflow Snapshot** persists a resolved train, tune, or evaluate **Workflow Config** for remote execution or later benchmark collection.
- **Resolved Workflow Hydration** loads **Resolved Workflow Snapshots** directly and does not run **Surface** resolution.
- **Config Group Loading** feeds **Surface** resolution and raw config display/edit paths through separate Interfaces.
- A **Problem Spec** can be selected by name or supplied inline by benchmark problem grids.
- An **Evaluation Config Snapshot** freezes evaluator config provenance for persisted artifact evaluation state without representing the whole evaluate Workflow Config.
- A **Benchmark** contains one or more **Benchmark Cases**.
- A **Benchmark Case** contains one or more **Benchmark Steps**.
- A **Benchmark Step** contributes one or more **Benchmark Plan Entries** through **Benchmark Plan Materialization**.
- **Benchmark Plan Materialization** resolves dependency-produced root ids before producing **Benchmark Plan Entries** with **Resolved Workflow Snapshots**.
- **Benchmark Plan Execution** creates a **Benchmark Run** first, then submits the exact persisted plan.
- A **Benchmark Run** records **Benchmark Plan Entries**, submissions, and one **Benchmark Collection Snapshot**.
- A **Benchmark Collection Snapshot** contains **Benchmark Result Records** for all expected collected evaluate steps after **Evaluation Execution Provenance** matches the submitted execution ref.
- The **Benchmark Result Index** projects **Benchmark Result Records** for query and CSV export.
- A **Storage Selector** resolves existing persisted roots through the catalog before consumers build paths.
- A **Root Handle** is the workflow-facing result of resolving a **Storage Selector** or produced-root identity.
- A **Produced Root Handle** is derived from **Producer Root Identity** and canonical storage layout.
- **Root Lifecycle** changes storage roots and keeps the catalog index current.
- **Corpus Assembly** consumes a block source and produces a dry-run plan or committed corpus root.
- **Corpus Assembly** uses **Corpus Capability Planning** for acquisition-window and refill policy.
- **Corpus Assembly** uses **Corpus Acquisition Stage** for staging, fulfillment, commit wiring, and cleanup.
- **Corpus Acquisition Stage** owns **Corpus Split Materialization** session lifecycle and split sequencing.
- An **Artifact Inference Context** trusts artifact manifest semantics, validates selected corpus compatibility, and prepares model scoring inputs.
- A **Temporal Dataset Preparation Interface** owns temporal sample selection, split assignment, scaler use, builder runtime metadata, and prepared dataset assembly.
- An **Action Space** is derived by an execution policy from a problem store and selected temporal samples.
- A **Training Runner** consumes prepared training data and produces fitted model state plus runtime training metrics.
- A **Batch Plan** is built by the **Training Runner** and inference paths after runtime memory budget is known, carrying the prepared **Action Space** into inputs, targets, and decode.
- A **Forward Runtime Plan** builds the warmup and final **Batch Plans** for inference and split-metric forward passes.
- A **Training Runtime Plan** gives the **Training Runner** final train/validation **Batch Plans**, a measured runtime context, one reusable prediction training state, and an **Evaluation Scoring Runtime Plan** for objective evaluator scoring.
- A **Training Fit Policy** is internal to the **Training Runner** and does not change model math or callback ownership.
- A **Decoded Result ABI** is produced by a prediction contract and accepted by evaluator contracts by decoded-result id.
- An **Objective Runtime** turns validation metrics or model-bound evaluator scoring into objective metrics for the **Training Runner**.
- A **Temporal Replay Runner** is shared by replay evaluator Adapters after prediction decoding and before **Temporal Accounting**.
- **Temporal Accounting** is shared by evaluator Adapters after they select temporal decision events.
- **Workflow Command Selection** builds typed **Workflow Selections** from operator options before config resolution.
- A **Benchmark Collection Resolver** consumes exact artifact/evaluation ids, the pulled artifact root, and the submitted execution record to produce a collected benchmark evaluation.
- An **Execution Session** is opened for one explicit execution target and used by submission, following, remote transfer, and benchmark collection.

## Example Dialogue

> **Dev:** "Should the benchmark create temporary YAML files for every lookback window?"
> **Domain expert:** "No. The benchmark builds a **Workflow Selection** with an inline **Problem Spec**, then config resolution produces the **Workflow Config**."

## Flagged Ambiguities

- "request" previously meant unresolved workflow intent. Use **Workflow Selection** for that concept.

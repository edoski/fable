# Concrete Storage Implementation

Storage keeps datasets, trained artifacts, tuning studies, and derived catalog indexes on disk. It uses typed roots: each root has one purpose and one SQLite state DB.

## Mental Model

There are three root kinds:

```text
corpus root   -> acquired canonical block data
study root    -> tuning manifest + Optuna state
artifact root -> trained model + manifest + training/evaluation state
```

The catalog is an index over roots. Root manifests and state DBs are the source of truth.

## On-Disk Layout

```text
outputs/
  .spice/catalog.sqlite
  corpora/{chain}/{corpus_id}/
    blocks/
    .spice/state.sqlite
  studies/{chain}/{study_id}/
    .spice/state.sqlite
  artifacts/{chain}/{artifact_id}/
    model.pt
    .spice/state.sqlite
```

Each root DB has `spice_meta` recording its root kind.

## SQLite State Engine

The engine creates the expected tables for a root kind and validates table shape. Column order is part of the state contract.

| Root kind | Main tables |
| --- | --- |
| corpus | `dataset_manifest`, `acquire_runs` |
| artifact | `artifact_manifest`, `training_summary`, `evaluation_summary` |
| study | `study_manifest` plus Optuna tables |

SQLite connections enable foreign keys, WAL, and busy timeout.

## Payload Codecs

`storage.payloads` owns generic SQLite payload stores plus strict persisted-payload record helpers. Root-local payload codecs are named `PayloadCodec` objects for corpus, study, and artifact state. Persistence modules call those codec objects directly, so the persisted payload ABI has one seam per record type. Root table payload records use Pydantic validation with forbidden extras and strict scalar handling; malformed persisted payloads raise `StateLayoutError`, but exact Pydantic wording is not storage API. Semantic payload codecs use TypeAdapter canonical round trips because they persist contract bundles rather than root table envelopes.

Corpus manifests and acquire-run records are Pydantic-native durable records, so `storage.corpus_codecs` only binds those records to the SQLite payload store seam. Artifact and study codecs keep explicit storage records where they flatten runtime summaries, rebuild owner configs through registry-aware coercers, or validate semantic projections.

## Deterministic IDs

IDs are content-derived:

| ID | Source |
| --- | --- |
| Corpus id | Chain, corpus name, corpus window. |
| Study id | Canonical study identity payload. |
| Artifact id | Canonical artifact identity payload. |

Hashes make storage paths stable for identical experiment identities.

## Corpus State

Corpus state stores one corpus manifest plus acquire-run history. The manifest records corpus id, corpus name, chain name, chain id, source requirements, and canonical block-corpus provenance. The block manifest records the requested timestamp/block range, observed coverage extent, compact validation status/issues, materialization outcome, and backing file count.

Acquire-run rows store provider identity, endpoint fingerprint, sizing facts, acquisition config snapshot, and RPC controller counters.

## Artifact State

Artifact state stores exact training provenance:

```text
artifact manifest
  -> exact configs
  -> feature graph fingerprint
  -> scaler
  -> sequence runtime metadata
  -> semantics
  -> split/training/model/prediction
```

Training state stores one compact summary with rows, split sizes, best epoch,
best validation total loss, and test total loss. Evaluation state stores multiple
summaries keyed by an evaluation storage id derived from evaluator config, delay,
and execution ref when present.

Artifact manifest codecs serialize the persisted Temporal Capability envelope, including compiler runtime metadata payloads. Temporal owns the runtime capability value and compiler metadata dispatch; storage owns the artifact manifest payload shape.

## Study State

Study state stores a SPICE study manifest and Optuna's RDB tables in the same SQLite file. The study manifest validates that resumed tuning belongs to the same study definition. Trial attributes store sampled params and best epoch. `storage.study_optuna` owns the RDB adapter and study read APIs; `modeling.tuning_execution` owns opening execution, running trials, and writing per-trial execution metadata.

## Operator Outcomes

`storage.operator` is the command/query Module for existing roots. It takes catalog selectors, decides whether a show request lists matches or renders one root, validates detail requests, returns ambiguity diagnostics with narrowing attributes, and converts blocked deletes into renderable dependent-root sections. It does not know CLI flag spelling; command modules map selector attributes such as `model_id` to flags such as `--model`.

## Staging And Commit

`storage.transactions` is the workflow-facing commit API. Public entrypoints are handle-shaped: corpus acquisition commits selected corpus paths from staged sources, training commits an artifact root through a writer callback, and tuning records study-root mutations. Full-root commits use hidden staged roots:

```text
writer receives staged root
  -> validate root kind
  -> atomically promote over destination
  -> reindex catalog
```

Corpus acquisition promotes selected paths inside a corpus root from the corpus handle. Existing-root mutation effects validate the expected root kind and reindex after successful mutation; tune uses this for study state, including study opening. Evaluation state is recorded through `storage.transactions.record_artifact_evaluation_state()` and intentionally does not reindex because artifact catalog rows derive from the manifest, not evaluation summaries.

`workflow_roots.py` exposes root handle models and read behavior. `workflow_root_materialization.py` resolves selectors, derives produced ids, materializes scalar root facts, and assembles workflow root sets; Workflow Preparation consumes those handles for preflight. Existing catalog records are identity/search facts only, so `storage.catalog.materialization` derives root paths and state DB paths from storage root plus record identity. Storage Transactions own handle-shaped promotion, selected-path commit, mutation, and reindex boundaries.

## Transfer Support

Supported transfer directions:

| Command | Direction |
| --- | --- |
| `transfer push dataset` | local corpus root to cluster. |
| `transfer pull artifact` | cluster artifact root to local. |

`execution.transfer_transaction` prepares a remote or local stage, uses `rsync`, validates root kind through lifecycle operations, finalizes the stage, and reindexes. `storage.sync_cli` is the remote-side helper for path and root-kind commands; finalize emits the promoted catalog record through the strict catalog codec. Transfer destination paths are derived by catalog materialization, and remote records cross the SSH seam through the strict catalog codec. Dataset is the operator identity; corpus is the storage root kind. Storage inspection owns local artifact dependency warnings, and CLI transfer commands render them.

## Invariants

| Rule | Why |
| --- | --- |
| Root kind must match operation. | Prevents reading a corpus DB as an artifact DB. |
| Table shape must match expected schema. | State payload decoders depend on columns. |
| Manifest is source of truth. | Catalog can be rebuilt. |
| Study contains exactly one Optuna study. | Trial identity stays unambiguous. |
| Artifact delay cannot exceed trained capability. | Prediction head width and problem geometry must align. |

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Missing state DB | Root was not materialized. |
| Root-kind mismatch | Wrong storage path passed to an operation. |
| Schema mismatch | State DB does not match expected tables. |
| Multiple selector matches | CLI/storage resolver needs a unique root. |
| Delete blocked by dependents | Dataset or study still has linked children. |
| Missing best params | Tuned artifact requested before study has a best trial. |

## Extension Pattern

New persisted state should be rooted in a manifest first, then indexed into the catalog through catalog materialization and store operations. Keep catalog rows derivable from root state.

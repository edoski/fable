# Concrete Storage Implementation

Storage keeps datasets, trained artifacts, tuning studies, and derived catalog indexes on disk. It uses typed roots: each root has one purpose and one SQLite state DB.

## Mental Model

There are three root kinds:

```text
corpus root   -> acquired history/evaluation block data
study root    -> tuning manifest + Optuna state
artifact root -> trained model + manifest + training/evaluation state
```

The catalog is an index over roots. Root manifests and state DBs are the source of truth.

## On-Disk Layout

```text
outputs/
  .spice/catalog.sqlite
  corpora/{chain}/{corpus_id}/
    history/
    evaluation/
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
| artifact | `artifact_manifest`, `training_summary`, `training_epochs`, `evaluation_summary`, `evaluation_runs` |
| study | `study_manifest` plus Optuna tables |

SQLite connections enable foreign keys, WAL, and busy timeout.

## Payload Codecs

`storage.payloads` owns generic SQLite payload stores plus raw persisted-payload validation helpers. Root-local payload codecs for corpus, study, and artifact state use strict `PayloadModel` envelopes at the SQLite boundary so malformed state consistently raises `StateLayoutError`. Semantic payloads use TypeAdapter canonical round trips because they persist dataclass contract bundles rather than root table envelopes.

## Deterministic IDs

IDs are content-derived:

| ID | Source |
| --- | --- |
| Corpus id | Chain, dataset name, evaluation date. |
| Study id | Canonical study identity payload. |
| Artifact id | Canonical artifact identity payload. |

Hashes make storage paths stable for identical experiment identities.

## Corpus State

Corpus state stores one dataset manifest plus acquire-run history. The manifest records dataset id, dataset name, chain name, chain id, and split-level history/evaluation provenance. Each split records the requested timestamp/block range, observed coverage, validation report, materialization outcome, and backing file count.

Acquire-run rows store provider identity, endpoint fingerprint, sizing facts, acquisition config snapshot, and RPC controller counters.

## Artifact State

Artifact state stores exact training provenance:

```text
artifact manifest
  -> exact configs
  -> feature graph fingerprint
  -> scaler
  -> builder runtime metadata
  -> semantics
  -> split/training/model/prediction/objective
```

Training state stores one summary and ordered epoch rows. Evaluation state stores multiple summaries keyed by an evaluation storage id derived from evaluator config, delay, and execution provenance when present.

## Study State

Study state stores a SPICE study manifest and Optuna's RDB tables in the same SQLite file. The study manifest validates that resumed tuning belongs to the same study definition. Trial attributes store sampled params and best epoch. `storage.study_optuna` owns the RDB adapter and study read APIs; `modeling.tuning_execution` owns opening execution, running trials, and writing per-trial execution metadata.

## Operator Outcomes

`storage.operator` is the command/query Module for existing roots. It takes catalog selectors, decides whether a show request lists matches or renders one root, validates detail requests, returns ambiguity diagnostics with narrowing attributes, and converts blocked deletes into renderable dependent-root sections. It does not know CLI flag spelling; command modules map selector attributes such as `model_id` to flags such as `--model`.

## Staging And Commit

`storage.transactions` is the workflow-facing commit API. Full-root commits use hidden staged roots:

```text
write staged root
  -> validate root kind
  -> atomically promote over destination
  -> reindex catalog
```

Partial commits promote selected paths inside an existing root. Acquire uses partial commit because history, evaluation, and state paths are assembled as parts of a corpus root.

`workflow_roots.py` only exposes root handles and catalog-record conversion. Workflow Preparation derives produced roots and resolves consumed roots; Storage Transactions own promotion, selected-path commit, and reindex boundaries.

## Transfer Support

Supported transfer directions:

| Command | Direction |
| --- | --- |
| `transfer push dataset` | local corpus root to cluster. |
| `transfer pull artifact` | cluster artifact root to local. |

`execution.transfer` prepares a remote or local stage, uses `rsync`, validates root kind through lifecycle operations, finalizes the stage, and reindexes. `storage.sync_cli` is the remote-side helper for path and root-kind commands. Transfer destination paths are derived by catalog materialization, and remote records cross the SSH seam through the strict catalog codec. Dataset is the operator identity; corpus is the storage root kind.

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

# Storage Architecture

## Purpose

`storage` owns deterministic identity, canonical path layout, root-local SQLite state, catalog records, root lifecycle mechanics, inspection, deletion, and the remote-side transfer helper.

ML artifacts need provenance. A model file without its dataset, features, temporal problem, prediction contract, evaluator config, and training config is hard to trust. Spice stores authoritative state inside each root and keeps a separate catalog for discovery.

## Root Layout

```text
outputs/
  .spice/catalog.sqlite
  corpora/<chain>/<corpus_id>/.spice/state.sqlite
  studies/<chain>/<study_id>/.spice/state.sqlite
  artifacts/<chain>/<artifact_id>/.spice/state.sqlite
```

Root-local state is authoritative. The catalog is an index that can be rebuilt by scanning roots.

## Root Kinds

```text
RootKind.CORPUS    canonical block data
RootKind.STUDY     tuning/search state
RootKind.ARTIFACT  trained model and evaluation state
```

Every state database stores root-kind metadata. Read APIs validate root kind before interpreting a database as a corpus, study, or artifact root. A table name alone is not sufficient proof that a database belongs to the expected domain.

Malformed persisted payloads raise `StateLayoutError`. Missing expected state raises `MissingStateError`.

## Commit Patterns

```text
train / transfer pull complete root
  write complete staged root
  -> validate staged root kind
  -> atomically promote root
  -> reindex catalog

acquire partial corpus update
  write staged history/evaluation/state files
  -> atomically replace selected paths
  -> reindex corpus root

tune existing study root
  mutate root-local search state
  -> validate study root kind
  -> reindex study root

evaluate existing artifact root
  record evaluation state
  -> keep catalog unchanged because artifact catalog rows derive from manifest
```

`storage.transactions` owns workflow-facing transaction objects. `PartialRootTransaction` always replaces selected paths. `FullRootTransaction.commit()` wraps complete-root staging, writer execution, promotion, cleanup, and reindex. Existing-root mutation helpers validate the expected root kind and reindex after successful mutation. `workflow_roots.py` carries root handles only; it does not own promotion, reindex, or evaluation-state mutation policy.

## Identity Rule

Ids hash provenance, not paths:

```text
semantic provenance -> canonical payload -> deterministic id
```

Output location should not change corpus, study, or artifact identity. Run controls that do not affect durable semantics, such as tuning trial limits, should not change study identity.

Catalog `created_at` is stable across upserts. `updated_at` changes when a record is refreshed.

## Module Map

```text
storage/
  layout.py          canonical paths
  identity.py        provenance payload assembly
  root_identity.py   pure consumed/produced root id/fact helpers
  ids.py             deterministic id hashing
  errors.py          storage-owned operator errors with storage record payloads
  engine.py          SQLite engine and root-kind metadata
  schema.py          root-local state schema
  selectors.py       typed catalog selectors
  payloads.py        generic payload stores/codecs
  corpus_codecs.py   corpus-root payload ABI
  artifact_codecs.py artifact-root payload ABI
  semantics_codecs.py persisted semantic-contract payload ABI
  corpus.py          corpus-root persistence
  study_manifest_codecs.py study-root manifest payload ABI
  study_manifest.py  study-root manifest persistence
  study_models.py    study runtime read models
  study_render.py    compact study result rendering helpers
  study_optuna.py    Optuna storage adapter and read access
  artifact.py        artifact-root persistence
  operator.py        show/delete command outcomes and ambiguity policy
  workflow_roots.py  workflow-facing root handle models and construction
  workflow_root_materialization.py  selector resolution and workflow root-set assembly
  transactions.py    workflow-facing root commit/mutation/reindex boundaries
  lifecycle.py       low-level staging, promotion, validation, and delete cascade
  sync_cli.py        remote-side transfer path/root-kind helper commands
  inspect*.py        read-only inspection views
  catalog/           global searchable index, catalog records, schema, store, and codecs
```

Storage owns persisted payload ABI. Modeling and evaluation own runtime result objects; named storage `PayloadCodec` objects translate those objects at the SQLite table boundary. Persistence modules call codec objects directly, keeping encode/decode locality at one seam per persisted record type. Artifact manifests persist Temporal Capability as the artifact-facing compiler capability bundle and persist artifact semantics as its normalized semantic projection. Artifact evaluation state stores an **Evaluation Config Snapshot**, not a live evaluator config object, so evaluation storage identity is based on immutable evaluator provenance.

Producer identity and consumer selection stay separate inside `storage.workflow_root_materialization`. That module resolves existing roots through the catalog, derives produced root ids, and assembles workflow root sets. `workflow_roots.py` owns all catalog-record and producer-identity handle construction, so layout/state-path rules for handles have one seam. Root handles expose root facts and manifest loading. Storage transactions expose handle-shaped staging and mutation helpers to workflows while keeping root-kind, prune, promotion, selected-path commit, existing-root mutation, and reindex policy inside storage; lower-level lifecycle remains path and root-kind infrastructure. Benchmark Plan Materialization uses storage root-fact helpers and catalog lookup through an internal adapter; benchmark ledger shape stays benchmark-owned.

`operator.py` owns Storage Operator Outcomes for show/delete command behavior: list-vs-detail selection, detail ambiguity, narrowing attributes, delete-blocked diagnostics, and refresh rendering. CLI code maps options to selectors, maps narrowing attributes to flag names, and prints renderable sections.

## Remote Transfer Boundary

Remote SSH and rsync orchestration lives in `execution.transfer_transaction`. Storage exposes local lifecycle/catalog operations and `storage.sync_cli`, the helper module executed on the remote machine for path and root-kind operations.

Catalog kind `dataset` maps intentionally to storage `RootKind.CORPUS`: dataset is the operator/config identity, while corpus is the physical storage root kind. `storage.catalog.index` owns typed catalog list/resolve/upsert/reindex operations, private catalog dispatch owns root-kind metadata, `storage.catalog.materialization` owns canonical destination paths and record construction, and `storage.catalog.codecs` owns the strict remote JSON envelope used by transfer helpers.

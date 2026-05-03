# Storage Architecture

## Purpose

`storage` owns deterministic identity, canonical path layout, root-local SQLite state, catalog records, root lifecycle mechanics, inspection, deletion, and the remote-side sync helper.

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

tune / evaluate existing roots
  mutate root-local state directly
  -> reindex only when search records/materialized state change
```

`PartialRootCommit` always replaces selected paths. Full root staging keeps explicit `replace` behavior because push/pull operations expose overwrite policy.

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
  study_optuna.py    Optuna storage adapter and read access
  artifact.py        artifact-root persistence
  operator.py        show/delete command outcomes and ambiguity policy
  workflow_roots.py  workflow-facing roots, producer identity, and root operations
  lifecycle.py       staging, promotion, partial commit, delete cascade
  sync_cli.py        remote-side path/root-kind helper commands
  inspect*.py        read-only inspection views
  catalog/           global searchable index, root-kind registry, record codecs
```

Storage owns persisted payload ABI. Modeling and evaluation own runtime result objects; storage codecs translate those objects at the SQLite table boundary. Artifact manifests persist Temporal Capability as the artifact-facing compiler capability bundle and persist artifact semantics as its normalized semantic projection. Artifact evaluation state stores an **Evaluation Config Snapshot**, not a live evaluator config object, so evaluation storage identity is based on immutable evaluator provenance.

Producer identity and consumer selection stay separate inside `workflow_roots.py`. Producer helpers derive ids and root handles for roots that workflows are about to create. Consumer helpers resolve existing roots through the catalog before workflows read them. Workflow roots expose storage-owned operations for manifest loading, staging, promotion, reindexing, and evaluation-state upsert; lower-level lifecycle remains path and root-kind infrastructure.

`operator.py` owns Storage Operator Outcomes for show/delete command behavior: list-vs-detail selection, detail ambiguity, narrowing attributes, delete-blocked diagnostics, and refresh rendering. CLI code maps options to selectors, maps narrowing attributes to flag names, and prints renderable sections.

## Remote Transfer Boundary

Remote SSH and rsync orchestration lives in `execution.transfer`. Storage exposes local lifecycle/catalog operations and `storage.sync_cli`, the helper module executed on the remote machine for path and root-kind operations.

Catalog kind `dataset` maps intentionally to storage `RootKind.CORPUS`: dataset is the operator/config identity, while corpus is the physical storage root kind. `storage.catalog.registry` owns root-kind dispatch for canonical destination paths, record resolution, record construction, codec field shape, catalog upsert, and catalog delete. `storage.catalog.codecs` owns the strict remote JSON envelope used by transfer helpers.

# Storage Catalog Architecture

## Purpose

`storage.catalog` contains searchable record models, relational catalog schema, private root-kind dispatch metadata, typed catalog read/write operations, and remote record codecs.

## Theory

The catalog is an index, not the source of truth. Root-local SQLite state owns provenance and runtime state. The catalog lets operators find roots by selectors without scanning every manifest for every command.

## Invariants

Catalog rows store identity and selector/search facts only. Root paths and state DB paths are materialized canonically from storage root plus catalog identity. Reindexing reads root-local state, verifies the scanned path matches manifest identity, and upserts catalog rows. Upserts must not rewrite `created_at`. Deleting catalog rows must be coordinated with root deletion in `storage.lifecycle`.

Private root-kind metadata names the table, primary key field, typed record, nullable fields, parent directory, and default ordering for each catalog root kind. Callers use `catalog.index` for list/resolve/upsert/reindex operations, `catalog.materialization` for canonical root locations, and `catalog.codecs` for remote record envelopes; they do not use registry specs directly.

## Extension Points

Add catalog columns only for stable selector/search fields. Do not store large runtime summaries here; keep those in root-local state.

Benchmark root ledgers are not catalog rows. They are benchmark-run audit state stored in benchmark plan/collection files and projected into the benchmark result index.

## Catalog Flow

```text
root-local state DB
        |
        v
catalog record materializer
        |
        v
private root-kind dispatch
        |
        v
catalog record
        |
        v
selector queries
```

## Table Shape

```text
dataset_index   dataset_id, dataset_name, chain_name, timestamps
study_index     study_id, dataset/model/problem selectors, timestamps
artifact_index  artifact_id, dataset/model/problem/variant selectors, timestamps
```

Catalog records are intentionally flat. Flat rows make selector queries simple and make CLI output predictable.

## Timestamp Rule

`created_at` means first catalog insertion time. `updated_at` means latest upsert time. Reindexing or refreshing can update `updated_at`, but it must not rewrite original creation time for the same primary key.

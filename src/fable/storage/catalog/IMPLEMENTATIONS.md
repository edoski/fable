# Concrete Catalog Implementation

The catalog is a fast index over storage roots. It helps CLI commands list, resolve, show, and delete datasets, studies, and artifacts.

## Mental Model

Catalog rows are derived summaries:

```text
root state DBs
  -> scan manifests
  -> upsert catalog rows
  -> selector queries
```

If the catalog is missing or stale, it can be refreshed from roots.

## Tables

| Table | Indexed root |
| --- | --- |
| `corpus_index` | Corpus roots. |
| `study_index` | Study roots. |
| `artifact_index` | Artifact roots. |

Rows store selector fields, created timestamp, and updated timestamp. Root locations are materialized from canonical layout rules instead of persisted in catalog rows. Upsert preserves `created_at` and changes `updated_at`.

## Refresh

`refresh_catalog()` rebuilds the catalog into a temporary DB, scans registered root-kind directories, detects root kind from each state DB, materializes the matching manifest into a typed record, upserts rows through catalog store operations, then atomically replaces the catalog DB.

```text
scan registered root-kind directories
  -> detect root kind
  -> materialize catalog record
  -> write temp catalog
  -> replace catalog.sqlite
```

## Single-Root Reindex

After a workflow promotes a root, `reindex_catalog_root()` updates the catalog entry for that root and returns the reindexed catalog record. It rejects scanned roots whose path does not match the manifest identity. This avoids full scans after every successful workflow.

`catalog.registry` is private metadata over corpus, study, and artifact roots. `catalog.store` owns SQLite row conversion and upsert/delete/list mechanics. `catalog.materialization` owns manifest-to-record conversion and canonical root locations. `catalog.codecs` owns the strict remote record envelope. `catalog.index` keeps the typed selector-facing API and is the public entrypoint for catalog reads, writes, refresh, and reindex.

## Selectors

Selectors filter by human-facing fields:

| Root | Main selector fields |
| --- | --- |
| Dataset | chain, dataset. |
| Study | chain, dataset, features, prediction, model, problem, study name. |
| Artifact | chain, dataset, features, prediction, model, problem, variant, study. |

Operations that mutate or show detailed state require exactly one match.

## Delete Safety

Delete validates that the selected path stays under the expected storage subtree and that root kind matches. Dataset delete checks dependent studies/artifacts; study delete checks dependent artifacts.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Zero matches | Selector does not identify an existing root. |
| Multiple matches | Selector is too broad. |
| Root path outside subtree | Materialized root location is unsafe for deletion. |
| Dependent roots exist | Delete needs explicit cascade. |
| Root-kind mismatch | Catalog row points to wrong root type. |

## Extension Pattern

Add catalog fields only when they help selectors or list views. Root manifests remain the complete provenance record.

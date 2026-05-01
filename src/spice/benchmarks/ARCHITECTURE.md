# Benchmark Architecture

## Purpose

`benchmarks` owns checked-in benchmark specs, workflow-selection expansion, dependency-aware plan materialization, durable run-state files, remote submission orchestration, all-or-nothing collection, SQLite indexing, read-only queries, and CSV export.

The benchmark domain should not live under config. It uses config resolution, but its own concepts are benchmark cases, steps, runs, submissions, collections, and result rows.

## Flow

```text
benchmark YAML spec
    |
    v
BenchmarkSpec
    |
    v
expanded Benchmark Workflow Selections
    |
    v
Benchmark Plan Materialization
    |
    v
durable run dir: metadata.json + plan.jsonl
    |
    v
submit persisted plan -> submission.jsonl
    |
    v
collect all expected evaluate results -> collection.json
    |
    v
results.sqlite projection -> CSV export/query
```

Planning builds typed workflow selections. **Benchmark Plan Materialization** asks storage-owned Dependency Root Materialization for dependency-produced root ids, calls normal workflow resolution, and produces plan entries with resolved workflow snapshots. Inline problem grids produce inline `ProblemSpec` values during planning; plan JSON stores the selected problem id, while the resolved workflow config stores the full executable problem.

## Module Map

```text
benchmarks/
  schema.py      benchmark YAML schema
  planning.py    dimension expansion and workflow-selection rows
  materialization.py  dependency-aware config resolution and plan entry assembly
  models.py      benchmark plan data models
  result_records.py  collection snapshot and result records
  result_store.py    low-level SQLite result projection
  result_index.py    index upsert/rebuild operations
  result_query.py    read-only result query interface
  __init__.py    benchmark API
  runs.py        run-state files and JSONL codecs
  submission.py  remote workflow submission service
  collection_resolver.py  remote evaluate-result resolver
  collection.py  remote result collection service
  ledger.py      CSV export adapter
```

## Boundaries

Run dirs are canonical benchmark audit state. `results.sqlite` is a rebuildable projection over `collection.json`; normalized observation and metric tables are the read model for list/export, while JSON payloads remain audit/debug payloads. `results.csv` is export-only and overwritten from the index.

The CLI creates run dirs, submits existing run dirs, collects existing run dirs, exports CSV, and reads the result index. It does not re-plan during submit or collect.

Remote transfer during collection goes through a **Benchmark Collection Resolver** using an **Execution Session** and `execution.transfer`. The resolver pulls the submitted artifact, consumes the returned local catalog record, and selects the matching evaluation summary without re-resolving current local catalog state. A collected evaluation must match the submitted `execution_ref`; stale artifact summaries from earlier jobs are rejected.

Benchmark JSON shapes are operator-facing. Keep them stable unless a field name is part of a deliberate terminology cleanup.

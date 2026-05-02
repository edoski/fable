# Benchmark Architecture

## Purpose

`benchmarks` owns checked-in benchmark specs, dependency-aware plan materialization, durable run-state files, remote submission orchestration, all-or-nothing collection, SQLite indexing, read-only queries, and CSV export.

The benchmark domain should not live under config. It uses config resolution, but its own concepts are benchmark cases, steps, runs, submissions, collections, and result rows.

## Flow

```text
benchmark YAML spec
    |
    v
Benchmark Plan Materialization
    |
    v
Benchmark Plan Entries
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

**Benchmark Plan Materialization** expands dimensions, asks the **Benchmark Dependency Ledger** to match local dependencies and `artifact_from` step references, asks the **Benchmark Root Ledger** to derive dependency-produced root ids, calls normal workflow resolution, and produces durable plan entries with resolved workflow snapshots. Inline problem grids produce inline `ProblemSpec` values during materialization; the selection ledger stores the selected problem id, while the resolved workflow config stores the full executable problem.

## Module Map

```text
benchmarks/
  schema.py      benchmark YAML schema
  materialization.py  spec expansion and plan-entry orchestration
  dependency_ledger.py  dependency graph validation and row matching
  root_ledger.py  consumed/produced root-id materialization
  selection_ledger.py  typed benchmark coordinate ledger
  models.py      benchmark plan data models
  result_records.py  collection snapshot and result records
  result_store.py    low-level SQLite result projection
  result_index.py    index upsert/rebuild/count/query operations
  __init__.py    benchmark API
  run_state_codec.py  benchmark-local metadata/plan/submission/collection codecs
  runs.py        run-dir lifecycle and public run-state facade
  submission.py  remote workflow submission service
  collection_resolver.py  remote evaluate-result resolver
  collection.py  remote result collection service
  ledger.py      CSV export adapter
```

## Boundaries

Run dirs are canonical benchmark audit state. `results.sqlite` is a rebuildable projection over `collection.json`; normalized observation and metric tables are the read model for list/export, while JSON payloads remain audit/debug payloads. CSV files are named export artifacts for concrete table, figure, appendix, or analysis inputs and are overwritten from the index.

`runs.py` is the public run-state facade. Benchmark run-state JSON/JSONL encoding stays benchmark-local and must not move into config or shared storage codecs.

The CLI creates run dirs, submits existing run dirs, collects existing run dirs, exports CSV, and reads the result index. It does not re-plan during submit or collect.

Remote transfer during collection goes through a **Benchmark Collection Resolver** using an **Execution Session** and `execution.transfer`. The resolver pulls the submitted artifact, consumes the returned local catalog record, and selects the matching evaluation summary without re-resolving current local catalog state. A collected evaluation must match the submitted `execution_ref`; stale artifact summaries from earlier jobs are rejected.

Benchmark JSON shapes are operator-facing. Keep them stable unless a field name is part of a deliberate terminology cleanup.

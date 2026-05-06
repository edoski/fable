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

**Benchmark Plan Materialization** expands dimensions, matches dependencies, normalizes dependency-derived root selection, calls normal workflow resolution, asks storage for root facts when catalog fallback is needed, and produces durable plan entries with resolved workflow snapshots. Inline problem grids produce inline `ProblemSpec` values during materialization; the selection ledger stores the selected problem id, while the resolved workflow config stores the full executable problem.

The public plan materialization seam is `benchmarks.plan_materialization`. Its private internals own case expansion, dependency matching, dependency-derived root selection, root finalization, and selection ledger materialization. Storage access is an internal adapter for resolved root facts and catalog fallback. Callers use `materialize_benchmark_plan()` and the durable models exported from that package; they do not import materialization internals.

## Root Facts And Ledger

Benchmark Root Facts are the caller-facing root identity interface on each plan entry. They expose the scalar consumed, produced, and artifact-source root ids needed by collection, result records, and run-state consumers without walking audit entries or re-reading workflow config.

The Benchmark Root Ledger is audit state, not storage catalog state and not the caller-facing read model. Each plan entry stores typed root ledger entries with `run_id`, workflow, role, root kind, root id, optional source run id, and root-specific ids. Roles are `consumed`, `produced`, and `source`; root kinds are `dataset`, `study`, and `artifact`.

Benchmark Plan Materialization owns the required order: prepare dependency-derived selection, resolve the workflow config, finalize root facts and the root ledger from resolved config identity, then record produced roots for later dependent steps. Tuned train steps can consume a study produced by a prior tune step. Evaluate steps can consume an artifact produced by a prior train step, while separately recording the artifact-source dataset.

## Module Map

```text
benchmarks/
  schema.py      benchmark YAML schema
  plan_materialization/  public plan materialization interface plus private internals
  result_records.py  collection snapshot and result records
  result_schema.py   SQLite result projection schema
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

Run dirs are canonical benchmark audit state. `results.sqlite` is a rebuildable projection over `collection.json`; normalized observation, metric, and root-ledger tables are the read model for list/export, while JSON payloads remain audit/debug payloads. CSV files are named export artifacts for concrete table, figure, appendix, or analysis inputs and are overwritten from the index.

`runs.py` is the public run-state facade. Benchmark run-state JSON/JSONL encoding stays benchmark-local and must not move into config or shared storage codecs.

The CLI creates run dirs, submits existing run dirs, collects existing run dirs, exports CSV, and reads the result index. It does not re-plan during submit or collect.

Remote transfer during collection uses an execution-owned `StorageTransferTransaction`; matching uses a **Benchmark Collection Resolver**. Collection builds a `BenchmarkCollectionSelection` from the plan entry and submission, asks the transaction for the selected local artifact record, then passes that record to the resolver. The resolver reads `artifact_record.state_db_path`, validates artifact and artifact-source dataset identity, matches `(evaluator_id, delay_seconds, execution_ref)`, rejects stale or missing execution provenance, and does not re-resolve the local catalog.

Benchmark JSON shapes are operator-facing. Keep them stable unless a field name is part of a deliberate terminology cleanup.

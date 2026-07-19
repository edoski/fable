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

**Benchmark Plan Materialization** expands dimensions, matches dependencies, normalizes dependency-derived root selection, calls normal workflow resolution, asks Storage Root Materialization for consumed/produced/source root facts, and produces durable plan entries with resolved workflow snapshots. Inline problem grids split workflow values from benchmark coordinates: the resolved workflow config stores the full executable `ProblemSpec`, while the selection ledger stores the selected problem id.

Benchmark schema owns benchmark-supported workflows and dimension names. Plan materialization owns root-vs-coordinate selection policy for the durable selection ledger. Config owns only generic workflow selection models and field introspection.

The public plan materialization seam is `benchmarks.plan_materialization`. Its private internals own case expansion, dependency matching, dependency-derived root selection, Benchmark Root Facts assembly, Benchmark Root Ledger assembly, and selection ledger materialization. Storage Root Materialization derives scalar storage root facts; benchmark owns the durable ledger/run-state shape. Callers use `materialize_benchmark_plan()` and the durable models exported from that package; they do not import materialization internals.

## Root Facts And Ledger

Benchmark Root Facts are the caller-facing root identity interface on each plan entry. They expose the scalar consumed, produced, and artifact-source root ids needed by collection, result records, and run-state consumers without walking audit entries or re-reading workflow config.

The Benchmark Root Ledger is audit state, not storage catalog state and not the caller-facing read model. Each plan entry stores typed root ledger entries with `run_id`, workflow, role, root kind, root id, optional source run id, and root-specific ids. Roles are `consumed`, `produced`, and `source`; root kinds are `dataset`, `study`, and `artifact`.

Benchmark Plan Materialization owns the required order: prepare dependency-derived selection, resolve the workflow config, finalize root facts and the root ledger from resolved config identity, then record produced root facts for later dependent steps. Tuned train steps can consume a study produced by a prior tune step. Evaluate steps can consume an artifact produced by a prior train step, while separately recording the artifact-source dataset.

## Module Map

```text
benchmarks/
  schema.py      benchmark YAML schema
  plan_materialization/  public plan materialization interface plus private internals
  result_records.py  collection snapshot and result records
  _result_schema.py  private SQLite result projection schema
  result_index.py    result-index upsert/rebuild/count/query/export Interface
  __init__.py    benchmark API
  _run_state_codec.py  private metadata/plan/submission/collection codecs
  runs.py        public run-state Interface and run-dir lifecycle
  submission.py  durable benchmark run planning plus remote workflow submission service
  collection_resolver.py  remote evaluate-result resolver
  collection.py  remote result collection service
```

## Boundaries

Run dirs are canonical benchmark audit state. `results.sqlite` is a rebuildable projection over `collection.json`; list and export consume the Benchmark Result Index row as the single read model, backed by normalized observation and metric tables. CSV files are named export artifacts for concrete table, figure, appendix, or analysis inputs and are overwritten from the index.

`runs.py` is the public run-state Interface. Benchmark run-state JSON/JSONL encoding stays benchmark-local and private to `_run_state_codec.py`; callers create, load, record submissions, and read/write collection snapshots through `runs.py`.

Run-state files have deliberate roles. `metadata.json` stores the benchmark name, creation time, and target. `plan.jsonl` stores one Benchmark Plan Entry per row: dependencies, dimension labels, selection ledger, Benchmark Root Facts, Benchmark Root Ledger, and a Resolved Workflow Snapshot. `submission.jsonl` stores one submitted workflow row per benchmark run id. `collection.json` stores schema-versioned all-or-nothing collection results for expected evaluate entries.

The CLI creates run dirs, submits existing run dirs, collects existing run dirs, exports CSV, and reads the result index. It does not re-plan during submit or collect.

Remote transfer during collection uses an execution-owned `StorageTransferTransaction`; matching uses a **Benchmark Collection Resolver**. Collection builds a `BenchmarkCollectionSelection` from the plan entry, submission, and benchmark target, asks the transaction for the selected local artifact record, then passes that record to the resolver. The resolver materializes the local artifact state path from storage root plus artifact identity, validates artifact and artifact corpus identity, matches evaluator id, resolved delay, and full execution provenance, returns Benchmark Collection Match Facts for result records, and does not re-resolve the local catalog.

Benchmark JSON shapes are operator-facing. Keep them stable unless a field name is part of a deliberate terminology change.

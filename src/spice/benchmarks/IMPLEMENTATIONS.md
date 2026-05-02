# Benchmark Implementations

## Benchmark Plan Materialization

`plan_benchmark()` is the external Interface for turning a named Benchmark into durable Benchmark Plan Entries. It resolves once. Submit and collect consume persisted run-state files and do not re-plan.

Materialization keeps three ledgers distinct:

- `BenchmarkDependencyLedger` owns matched local run ids, external Slurm dependencies, and the `artifact_from` source run id.
- `BenchmarkSelectionLedger` owns benchmark coordinate intent such as surface, chain, model, problem, objective, evaluation, runtime knobs, and inline problem ids. It does not carry consumed root ids.
- `BenchmarkRootLedger` owns consumed root ids, produced root ids, `artifact_from` identity, and the artifact-source dataset id used to derive produced artifact ids.

The root ledger owns dependency-derived root policy. Tuned train steps without an explicit `study_id` consume the produced study id from a prior tune dependency. Evaluate steps with `artifact_from` consume the produced artifact id from the referenced train step. Evaluate dataset selection stays explicit when provided; otherwise it inherits the artifact source dataset. Explicit tuned train studies still resolve their dataset through the storage catalog.

`plan.jsonl` stores the typed ledgers plus a Resolved Workflow Snapshot. Raw JSON validation stays in `run_state_codec.py`; materialization works with typed benchmark and workflow objects.

## Result Index

Benchmark run dirs remain the audit source of truth. `results.sqlite` is a rebuildable projection over `collection.json`.

Collection snapshots copy the typed dependency, selection, and root ledgers from the plan entry. Result index rows read normalized coordinates from typed fields, not from raw payload JSON. Artifact dataset identity and evaluation dataset identity are stored separately so cross-corpus evaluation remains inspectable.

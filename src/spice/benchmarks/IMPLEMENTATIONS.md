# Benchmark Implementations

## Benchmark Planning

`benchmarks.plan_materialization.materialize_benchmark_plan()` turns a named benchmark into durable benchmark plan entries. It resolves once. Submit and collect consume persisted run-state files and do not re-plan.

Materialization keeps three durable ledgers distinct:

- `BenchmarkDependencyLedger` owns matched local run ids, external Slurm dependencies, and the `artifact_from` source run id.
- `BenchmarkSelectionLedger` owns benchmark coordinate intent such as surface, chain, model, problem, objective, evaluation, runtime knobs, and inline problem ids. It does not carry consumed root ids.
- `BenchmarkRootFacts` owns caller-facing consumed, produced, and source root ids.
- `BenchmarkRootLedger` owns typed audit entries for consumed, produced, and source roots.

Benchmark Plan Materialization owns the dependency/root sequence before the plan entry is assembled. It resolves dependency ledgers, prepares dependency-derived selections, finalizes root facts and the root ledger from the resolved workflow config, and records produced roots for later dependent steps. Tuned train steps without an explicit `study_id` consume the produced study id from a prior tune dependency. Evaluate steps with `artifact_from` consume the produced artifact id from the referenced train step. Evaluate dataset selection stays explicit when provided; otherwise it inherits the artifact source dataset. Explicit tuned train studies resolve their dataset through materialization's storage fact adapter. The persisted plan fields are `root_facts` for caller reads and `root_ledger` for audit entries.

`plan.jsonl` stores the typed ledgers plus a Resolved Workflow Snapshot. Raw JSON validation stays in `run_state_codec.py`; materialization works with typed benchmark and workflow objects.

## Result Index

Benchmark run dirs remain the audit source of truth. `results.sqlite` is a rebuildable projection over `collection.json`.

Collection snapshots copy the typed dependency, selection, and root ledger from the plan entry. Result records consume Benchmark Root Facts for artifact, artifact dataset, and evaluation dataset identity. Result index rows read normalized coordinates from typed fields, not from raw payload JSON. Artifact dataset identity and evaluation dataset identity are stored separately so cross-corpus evaluation remains inspectable. The index also projects `benchmark_root_ledger` rows keyed by observation so root audit state can be queried without decoding payload JSON.

## Collection Resolver

Collection selection is explicit. `BenchmarkCollectionSelection` is built from one evaluate plan entry and its submission record, validates run id, workflow, config, and Benchmark Root Facts consistency, and carries the artifact id, evaluation dataset id, artifact-source dataset id, evaluator id, configured delay, storage root, and submitted execution ref.

The resolver consumes a local artifact catalog record. It validates that the record and loaded artifact manifest match the selected artifact id and source dataset, then matches evaluation summaries by evaluator id, resolved delay, and exact execution provenance. Missing matching summaries return `None` so collection can fail all-or-nothing; stale or duplicate provenance raises an operator error.

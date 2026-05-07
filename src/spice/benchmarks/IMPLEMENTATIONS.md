# Benchmark Implementations

## Benchmark Planning

`benchmarks.plan_materialization.materialize_benchmark_plan()` turns a named benchmark into durable benchmark plan entries. It resolves once. Submit and collect consume persisted run-state files and do not re-plan.

Materialization keeps three durable ledgers plus one root-facts read model distinct:

- `BenchmarkDependencyLedger` owns matched local run ids, external Slurm dependencies, and the `artifact_from` source run id.
- `BenchmarkSelectionLedger` owns benchmark coordinate intent such as surface, chain, model, problem, objective, evaluation, runtime knobs, and inline problem ids. It does not carry consumed root ids.
- `BenchmarkRootFacts` owns caller-facing consumed, produced, and source root ids.
- `BenchmarkRootLedger` owns typed audit entries for consumed, produced, and source roots.

Benchmark Plan Materialization owns the dependency/root sequence before the plan entry is assembled. It resolves dependency ledgers, prepares dependency-derived selections, finalizes root facts and the root ledger from the resolved workflow config, and records produced root facts for later dependent steps. Tuned train steps without an explicit `study_id` consume the produced study id from a prior tune dependency. Evaluate steps with `artifact_from` consume the produced artifact id from the referenced train step. Evaluate dataset selection stays explicit when provided; otherwise it inherits the artifact source dataset. Explicit tuned train studies resolve their dataset through Storage Root Materialization. The persisted plan fields are `root_facts` for caller reads and `root_ledger` for audit entries.

Problem-grid expansion keeps two rows per seed: the workflow row can carry an inline executable `ProblemSpec`, while the selection row carries the stable problem id used by the benchmark selection ledger. Normal dimensions write the same value to both rows.

`plan.jsonl` stores the typed ledgers plus a Resolved Workflow Snapshot. `_run_state_codec.py` reads persisted JSON and JSONL through strict Pydantic adapters, then hydrates workflow config snapshots through the config-owned Resolved Workflow Hydration seam. Materialization works with typed benchmark and workflow objects through the `runs.py` run-state Interface.

## Result Index

Benchmark run dirs remain the audit source of truth. `results.sqlite` is a rebuildable projection over `collection.json`. `benchmarks.result_index` owns the public Result Index Interface: collection upsert, rebuild, counts, filtered list rows, and CSV export.

Collection snapshots copy the typed dependency, selection, and root facts from the plan entry. Result records consume Benchmark Collection Match Facts for artifact, artifact dataset, evaluation dataset, evaluation storage, evaluator, delay, and execution provenance identity. The Benchmark Result Index row is the list/export read model and reads normalized coordinates from typed fields. Artifact dataset identity and evaluation dataset identity are stored separately so cross-corpus evaluation remains inspectable. Root audit state stays in plan run-state; collection snapshots and the result index store scalar root facts needed for reads.

## Collection Resolver

Collection selection is explicit. `BenchmarkCollectionSelection` is built from one evaluate plan entry, its submission record, and the benchmark target. It validates run id, workflow, config, and Benchmark Root Facts consistency, and carries the artifact id, artifact dataset id, evaluation dataset id, evaluator id, configured delay, storage root, submitted execution ref, job id, log path, workflow task, and target.

The resolver consumes a local artifact catalog record. It validates that the record and loaded artifact manifest match the selected artifact id and artifact dataset, then matches evaluation summaries by evaluator id, resolved delay, and exact execution provenance. Missing matching summaries return `None` so collection can fail all-or-nothing; stale or duplicate provenance raises an operator error. A resolved match returns `BenchmarkCollectionMatchFacts`, which result records consume instead of rebuilding match identity from summaries or root audit entries.

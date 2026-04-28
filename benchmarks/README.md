# Benchmark Results

`results.csv` is the committed ledger for current benchmark results. One row is one
completed artifact evaluation under one evaluator and delay.

The ledger records modern evaluator results only. Current result rows use
`evaluation=poisson_replay_2h_mean`; historical `paper_replay_2h` results stay in
notes such as `PROGRESS.md` until rerun under current semantics.

## Row Contract

Required provenance fields:

- `recorded_at_utc`: UTC time the result row was recorded.
- `git_commit`: commit used for the run.
- `execution_ref`: `slurm:<job_id>` for remote runs or `local:<timestamp>` for local runs.
- `artifact_id`: persisted artifact identity.
- `evaluation_storage_id`: persisted evaluation identity inside the artifact state.

Required evaluator metric columns:

- `profit_over_baseline`
- `cost_over_optimum`
- `baseline_cost_over_optimum`

Optional existing ML metric columns:

- `total_loss`
- `offset_accuracy`
- `macro_f1`
- `classification_loss`
- `regression_loss`
- `log_fee_mae`
- `log_fee_mse`
- `exact_optimum_hit_rate`

`log_fee_mae` and `log_fee_mse` are measured in unnormalized log-fee units. They are
prediction-family diagnostics, not raw-fee economic metrics.

Blank optional metric cells mean the metric was not collected for that prediction family or
result. They do not mean zero.

## Sweep And Collection Context

`spice benchmark plan <name>` expands YAML benchmark specs from
`src/spice/conf/benchmark/` into JSONL workflow steps.

`spice benchmark submit <name>` submits the plan to the default remote target and
creates local run state under `outputs/benchmarks/runs/<name>/<timestamp>/`:

- `plan.jsonl`: resolved workflow snapshots.
- `submission.jsonl`: Slurm job ids, `execution_ref`, remote git commit, and logs.
- `collections/*.jsonl`: collection attempts and row status.

`spice benchmark collect <name>` reads the latest run directory, pulls completed
remote studies/artifacts through the existing storage sync APIs, and prints JSONL
collection status. Re-run it safely while jobs are still finishing. With `--write`,
collection appends only complete, non-duplicate rows to this ledger; missing expected
evaluation rows abort the write.

Raw HPO grids belong in `src/spice/conf/tuning_space/*.yaml`.

This ledger does not define sweeps, add tuning fields, or store raw artifacts. Runtime
artifact SQLite remains the detailed machine record.

# Benchmark Results

Benchmark run directories are the audit source of truth. `results.sqlite` is a
rebuildable query index over completed run snapshots. `results.csv` is an explicit
human-readable export from that index, not durable state.

The ledger records modern evaluator results only. Current result rows may use
`evaluation=poisson_replay_2h` or `evaluation=full_temporal_replay`; historical
`paper_replay_2h` results stay in notes such as `PROGRESS.md` until rerun under
current semantics.

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
- `exact_optimum_hit_rate`

Optional existing ML metric columns:

- `total_loss`
- `offset_accuracy`
- `macro_f1`
- `classification_loss`
- `regression_loss`
- `log_fee_mae`
- `log_fee_mse`

`log_fee_mae` and `log_fee_mse` are measured in unnormalized log-fee units. They are
prediction-family diagnostics, not raw-fee economic metrics.

Blank optional metric cells mean the metric was not collected for that prediction family or
result. They do not mean zero.

## Sweep And Collection Context

`spice benchmark plan <name>` expands YAML benchmark specs from
`src/spice/conf/benchmark/` into a durable run directory:

- `metadata.json`: benchmark name, target, creation time.
- `plan.jsonl`: resolved workflow snapshots.
- `submission.jsonl`: Slurm job ids, `execution_ref`, remote git commit, and logs.
- `collection.json`: complete collection snapshot, written only after every expected
  evaluate result is found.

Operator flow:

```bash
spice benchmark plan lookback_window_sweep --target disi_l40
spice benchmark submit outputs/benchmarks/runs/lookback_window_sweep/<timestamp>
spice benchmark collect outputs/benchmarks/runs/lookback_window_sweep/<timestamp>
spice benchmark index export --output benchmarks/results.csv
spice benchmark index rebuild
spice benchmark index list --benchmark lookback_window_sweep
```

Collection is all-or-nothing. If an expected evaluation summary is missing,
ambiguous, or does not match the submitted `execution_ref`, collection raises and
writes neither `collection.json` nor index rows.

Raw HPO grids belong in `src/spice/conf/tuning_space/*.yaml`.

Benchmark result records store summary-level metrics and provenance only. Raw replay
events, prediction tensors, decoded offsets, labels, and training epoch history stay in
artifact state or remote logs.

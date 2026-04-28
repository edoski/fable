# Concrete Config Presets

Checked-in YAML presets define runnable experiments without Python edits. A surface is the one-off workflow preset; benchmarks are matrix/DAG wrappers over surfaces and overrides. Presets are current runnable architecture only; historical unsafe and estimated-block paths are documented in top-level `ARCHIVE.md`.

## Default Surface

`current_row_fee_dynamics` resolves:

```yaml
chain: ethereum
dataset: icdcs_2026
features: core_fee_dynamics
problem: current_row_nominal
dataset_builder: fixed_sequence_temporal
model: lstm
prediction: icdcs_2026
objective: profit_poisson_replay_2h_mean
acquisition: {provider: publicnode}
training: {id: default, split: default}
tuning: {id: default, space: lstm_fixed_context}
evaluation: {id: poisson_replay_2h_mean}
```

Requests may override surface fields by name. `evaluation.delay_seconds` is optional in the surface; resolution defaults it to `problem.max_delay_seconds` for evaluation workflows.

The surface YAML uses targeted nesting where the nesting matches ownership:

| Field | Meaning |
| --- | --- |
| `chain` | Chain preset used for runtime chain id, POA handling, and nominal block time. |
| `dataset` | Corpus/evaluation-date preset. |
| `features` | Feature catalog/output preset. |
| `problem` | Temporal problem preset. |
| `dataset_builder` | Training/evaluation dataset builder. |
| `model` | Model family config. |
| `prediction` | Prediction head/decoder semantics. |
| `objective` | Training/tuning objective. |
| `acquisition.provider` | RPC provider preset. Provider YAML owns endpoint and acquisition runtime settings. |
| `training.id` | Training hyperparameter preset. |
| `training.split` | Split preset. |
| `tuning.id` | Optuna runtime preset. |
| `tuning.space` | Tuning search-space preset. |
| `evaluation.id` | Evaluator preset. |
| `evaluation.delay_seconds` | Concrete evaluation delay; defaults to problem capability when omitted. |

## Current Presets

Chains: `ethereum`, `polygon`, `avalanche`.

Provider: `publicnode`.

Dataset: `icdcs_2026`.

Features: `core_fee_dynamics`. `core_fee_dynamics_elapsed_position` is a post-refactor ablation preset only; it is identical to `core_fee_dynamics` plus `elapsed_seconds`, a corpus-position signal.

`core_fee_dynamics` selects safe fee, gas-pressure, cadence/calendar, rolling log-fee, tx-count, and fee-history priority-fee outputs. It does not include elapsed-time/corpus-position outputs and does not expose raw block author/proposer metadata.

Problems:

- `current_row_nominal`: `observed_time_window` with `slot_spacing.id: nominal`.
- `current_row_recent_median`: same compiler with `slot_spacing.id: recent_median`.

Dataset builders:

- `fixed_sequence_temporal`: derives and persists one fixed context length from training data.

Evaluators: `fullset`, `poisson_replay_2h_mean`, `poisson_replay_2h_total`, `zero_stop_rollout_fullset`, `anchor_basefee_fullset`.

Benchmarks: `safe_baseline_grid`, `large_capacity_hpo`, `lookback_window_sweep`, `slot_spacing_sweep`, `elapsed_position_ablation`, and `delay_degradation_sweep`.

`safe_baseline_grid` is the untuned ETH/POL/AVAX by LSTM/Transformer/Transformer-LSTM baseline. `large_capacity_hpo` is the bounded calibration search: the same 3x3 grid, large-capacity spaces, and 40 trials per cell. `lookback_window_sweep`, `slot_spacing_sweep`, `elapsed_position_ablation`, and `delay_degradation_sweep` are fixed train/evaluate grids, not per-cell HPO grids. `delay_degradation_sweep` trains one artifact per `max_delay_seconds` value and evaluates with the same delay through the default `evaluation.delay_seconds = problem.max_delay_seconds` resolution. Sample-count sweeps are deferred because larger history windows need explicit date-range and protocol-regime checks.

## Invariants

Config ids name concrete specs. Surface fields point at existing presets. Objective benchmark ids must match training/tuning evaluation configs. Old unsafe and estimated-block runnable paths are archived, not available as current configs.

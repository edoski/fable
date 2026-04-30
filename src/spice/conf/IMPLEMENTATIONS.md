# Concrete Config Specs

Checked-in YAML specs define runnable experiments without Python edits. A surface is the named workflow composition; benchmarks are matrix/DAG wrappers over surfaces and overrides. These specs are current runnable architecture only; historical unsafe and estimated-block paths are documented in top-level `ARCHIVE.md`.

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
objective: profit_poisson_replay_2h
acquisition: {provider: publicnode}
training: {id: default, split: default}
tuning: {id: default, space: lstm_fixed_context}
evaluation: {id: poisson_replay_2h}
```

Producer workflow selections may override surface fields by name. Evaluation is a root consumer: it selects an existing artifact and corpus by id, plus evaluator/runtime controls.

The surface YAML uses targeted nesting where the nesting matches ownership:

| Field | Meaning |
| --- | --- |
| `chain` | Chain spec used for runtime chain id, POA handling, and nominal block time. |
| `dataset` | Corpus/evaluation-date spec. |
| `features` | Feature catalog/output spec. |
| `problem` | Temporal problem spec. |
| `dataset_builder` | Training/evaluation dataset builder. |
| `model` | Model family config. |
| `prediction` | Prediction head/decoder semantics. |
| `objective` | Training/tuning objective. |
| `acquisition.provider` | RPC provider spec. Provider YAML owns endpoint and acquisition runtime settings. |
| `training.id` | Training hyperparameter spec. |
| `training.split` | Split spec. |
| `tuning.id` | Optuna runtime spec. |
| `tuning.space` | Tuning search-space spec. |
| `evaluation.id` | Evaluator spec used by train/tune objectives. |

## Current Specs

Chains: `ethereum`, `polygon`, `avalanche`.

Provider: `publicnode`.

Dataset: `icdcs_2026`.

Features: `core_fee_dynamics`. `core_fee_dynamics_elapsed_position` is a post-refactor ablation spec only; it is identical to `core_fee_dynamics` plus `elapsed_seconds`, a corpus-position signal.

`core_fee_dynamics` selects safe fee, gas-pressure, local trend, cadence/calendar, rolling log-fee, tx-count, and fee-history priority-fee outputs. It does not include elapsed-time/corpus-position outputs and does not expose raw block author/proposer metadata.

Problems:

- `current_row_nominal`: `observed_time_window` with `slot_spacing.id: nominal`.
- `current_row_recent_median`: same compiler with `slot_spacing.id: recent_median`.

Dataset builders:

- `fixed_sequence_temporal`: derives and persists one fixed context length from training data.

Evaluators: `poisson_replay_2h` and `full_temporal_replay`. The default surface uses `poisson_replay_2h`; `full_temporal_replay` is available as a sibling evaluator and objective.

Benchmarks: `safe_baseline_grid`, `large_capacity_hpo`, `lookback_window_sweep`, `slot_spacing_sweep`, `elapsed_position_ablation`, `delay_degradation_sweep`, and `evaluator_objective_grid`.

`safe_baseline_grid` is the untuned ETH/POL/AVAX by LSTM/Transformer/Transformer-LSTM baseline. `large_capacity_hpo` is the bounded calibration search: the same 3x3 grid, large-capacity spaces, and 40 trials per cell. `lookback_window_sweep`, `slot_spacing_sweep`, `elapsed_position_ablation`, and `delay_degradation_sweep` are fixed train/evaluate grids, not per-cell HPO grids. `delay_degradation_sweep` trains one artifact per `max_delay_seconds` value and evaluates with the artifact capability delay unless an evaluate step sets `delay_seconds`. Sample-count sweeps are deferred because larger history windows need explicit date-range and protocol-regime checks.

## Invariants

Config ids name concrete specs. Surface fields point at existing specs. Objective benchmark ids must match training/tuning evaluation configs. Old unsafe and estimated-block runnable paths are archived, not available as current configs.

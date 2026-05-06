# Archive

This file preserves historical runnable paths removed by the clean safe feature
refactor. These paths are not part of the current architecture.

Current runnable code uses evaluator `poisson_replay` and objective
`profit_poisson_replay`. Older evaluator ids below are historical only.

## Unsafe Same-Block Reference

Former runnable names included `same_block_closed` surfaces and
`same_block_closed_*` feature configs.

What it did: this path trained and evaluated a current-row decision model while
also exposing finalized facts from the same block used as the decision row. The
model could see same-block gas usage and related post-close information before
choosing a fee for that block.

Why it existed: it was useful as a professor-like comparator because it
reproduced the stronger baseline behavior observed in earlier experiments. It
helped separate "the model cannot match the paper" from "the paper-like setup is
using information that is not available at decision time."

Why it was removed: it is not a deployable current-row decision setting. The
decision row contains information that is only known after the block closes, so
keeping it runnable would encode data leakage into the architecture.

Progress context: `PROGRESS.md` keeps the completed delay-sensitivity,
checkpoint-selection, and targeted HPO result summaries that use this historical
reference. Those rows are thesis evidence, not current config targets.

Replacement: current runnable architecture uses `current_row_fee_dynamics` with
causal `core_fee_dynamics` features and the `current_row_nominal` problem. The
current base fee remains available only because EIP-1559 base fee for block `t`
is deterministic from parent state before block `t` execution; finalized
current-block facts are lagged.

## Estimated Block Compiler

Former runnable names included `estimated_block` and `estimated_block_*`
problem configs.

What it did: this compiler converted time horizons into estimated block-step
counts before constructing context and candidate windows. It expressed a
deadline as a block-count grid rather than using observed timestamps directly.

Why it existed: it supported paper-style nominal-grid experiments and comparator
sweeps, where a fixed expected block interval defined the prediction head width.

Why it was removed: it duplicated the modern timestamp compiler and made the
problem surface harder to reason about. Deadline windows are now defined from
observed timestamps, while `slot_spacing` only fixes the output head width.

Progress context: `PROGRESS.md` keeps historical references to paper-style
nominal-grid and interval-estimator experiments. Those references should not be
converted into runnable configs.

Replacement: current runnable architecture uses `observed_time_window` with
`slot_spacing.id: nominal` by default. `slot_spacing.id: recent_median` remains
available only as an explicit slot-spacing comparison problem.

## Variable Sequence Temporal Builder

Former runnable names included `variable_sequence_temporal` dataset-builder
configs and runtime metadata.

What it did: this builder let compiler-derived context start rows define
per-sample sequence length. It selected the tail `sample_count` valid anchors,
split them chronologically, and persisted compiler runtime metadata.

Why it existed: it was the first generic temporal dataset builder and matched
the variable lookback geometry emitted by the problem compiler.

Why it was removed: the thesis benchmark path now uses one fixed context length
derived from training data. Keeping both builders runnable preserved two dataset
semantics for the same benchmark surface without adding current experimental
value.

Replacement: current runnable architecture uses `fixed_sequence_temporal`, which
persists the calibrated sequence length, median block cadence, bounds, and
compiler metadata.

## Candidate Offset Selection Prediction Family

Former runnable names included `candidate_offset_selection` prediction configs
and the candidate-logit-only prediction family.

What it did: this family trained a softmax policy over candidate offsets using
expected fee cost and decoded the masked argmax offset.

Why it existed: it was a compact experimental offset-selection target useful for
early model and evaluator development.

Why it was removed: current thesis and paper-aligned work uses the ICDCS 2026
min-block-fee multitask target. Keeping the older family runnable duplicated the
prediction surface and pulled extra target/loss/metric code into the current
architecture.

Replacement: current runnable architecture uses `icdcs_2026`, backed by
`min_block_fee_multitask`, with offset classification plus min-log-fee
regression.

## Removed Benchmark Matrices

The clean refactor removed historical benchmark YAMLs that depended on unsafe or
estimated-block runnable paths. These matrices are archived here as historical
evidence only; they are not current config targets.

### `estimated_block_current_row_sweep`

Study: `estimated_block_current_row_sweep`.

Data grid:

- `same_block_closed` on `ethereum`, `polygon`, `avalanche`
- `block_open_lagged` on `ethereum`, `polygon`, `avalanche`

Model grid:

- `lstm` with `lstm_large_capacity`
- `transformer` with `transformer_large_capacity`
- `transformer_lstm` with `transformer_lstm_large_capacity`

Problem grid:

- `estimated_block_nominal_window`
- `current_row_nominal_window`
- `current_row_recent_delta_window`

Steps:

- `tune`: `validation_total_loss`, `fullset`, `trial_count: 120`
- `train_tuned`: after `tune`, `validation_total_loss`, `fullset`, `variant: tuned`
- `evaluate_tuned`: after `train_tuned`, scoring with `profit_poisson_replay_2h_mean` /
  `poisson_replay_2h_mean` and `profit_poisson_replay_2h_total` /
  `poisson_replay_2h_total`, using `variant: tuned`, `delay_seconds: 36`

### `current_row_problem_family_sweep`

Study: `current_row_problem_family_sweep`.

Data grid:

- `same_block_closed` on `ethereum`, `polygon`, `avalanche`
- `block_open_lagged` on `ethereum`, `polygon`, `avalanche`

Model grid:

- `lstm` with `lstm_large_capacity`
- `transformer` with `transformer_large_capacity`
- `transformer_lstm` with `transformer_lstm_large_capacity`

Problem grid:

- `current_row_nominal_window`
- `current_row_recent_delta_window`

Steps:

- `tune`: `validation_total_loss`, `fullset`, `trial_count: 120`
- `train_tuned`: after `tune`, `validation_total_loss`, `fullset`, `variant: tuned`
- `evaluate_tuned`: after `train_tuned`, scoring with `profit_poisson_replay_2h_mean` /
  `poisson_replay_2h_mean` and `profit_poisson_replay_2h_total` /
  `poisson_replay_2h_total`, using `variant: tuned`, `delay_seconds: 36`

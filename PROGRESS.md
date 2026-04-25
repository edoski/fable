# Progress

_Last verified: 2026-04-25 12:36 CEST_

## Status Snapshot

Local `main` is ahead of the remote experiment branch. That split is intentional: remote jobs are still running on `codex/temporal-parity`, so local cleanup commits should not be pushed into that environment until the current remote wave finishes.

The current local architecture names are `same_block_closed`, `block_open_lagged`, `current_row_nominal_window`, `poisson_replay_2h_mean`, and `poisson_replay_2h_total`. Remote benchmark logs still use older preset and evaluator names such as `icdcs_2026_oracle_intermediate`, `icdcs_2026_professor_block_open_*`, and `paper_replay_2h`.

`safe_best` is a historical benchmark role, not a local runnable config. It means the best family-specific safe block-open choices found before the current cleanup: LSTM with no broad time features, Transformer without `time_since_start`, and Transformer-LSTM with calendar-only time plus `recent_deltas`.

As of this verification, the fully completed baseline evidence is the delay-sensitivity sweep plus checkpoint-selection parity. The targeted `safe_best` HPO wave is not complete until Avalanche Transformer-LSTM job `57549` finishes and its dependent train/eval jobs `57550` and `57551` complete.

## Benchmarking

### Current Benchmark Context

- Unsafe reference surface: `same_block_closed`.
- Safe current-block surface: `block_open_lagged`.
- Current-row problem family: `current_row_nominal_window`.
- Explicit paper-style nominal-grid compiler path: `estimated_block`.
- Primary current evaluator: `poisson_replay_2h_mean`, reporting mean per-prediction `profit_over_baseline` and `cost_over_optimum`.
- Diagnostic total-ratio evaluator: `poisson_replay_2h_total`.
- Diagnostic fullset evaluators: `zero_stop_rollout_fullset` and `anchor_basefee_fullset`.

Historical remote results below use the older `paper_replay_2h` total-ratio style unless stated otherwise. Do not silently compare those numbers against current `poisson_replay_2h_mean` output.

### Active Remote Runs

Remote host: `edoardo.galli3@giano.cs.unibo.it`, storage root `/scratch.hpc/edoardo.galli3/spice/outputs`, log root `/scratch.hpc/edoardo.galli3/slurm`.

| Job | State | Role | Evidence |
| --- | --- | --- | --- |
| `57549` | Running | Avalanche Transformer-LSTM `safe_best` HPO to 40 trials | `spice-tune-57549.out`, best so far `0.0236` at trial 17 |
| `57550` | Pending | Train tuned Avalanche Transformer-LSTM after `57549` | dependency `afterok:57549` |
| `57551` | Pending | Evaluate tuned Avalanche Transformer-LSTM after `57550` | dependency `afterok:57550` |

Completed targeted HPO chains:

| Cell | HPO Result | Tuned Eval |
| --- | --- | --- |
| Ethereum LSTM | best `0.0140` | `0.0112` |
| Ethereum Transformer | best `0.0137` | `0.0112` |
| Ethereum Transformer-LSTM | best `0.0141` | `0.0115` |
| Avalanche Transformer | best `0.0224` | `0.0122` |
| Avalanche Transformer-LSTM | running, best so far `0.0236` | pending |

The HPO values are tuning/validation objective values. The tuned eval values are held-out `paper_replay_2h` results at `36s`.

### Last Verified Results

Delay-sensitivity sweep, `paper_replay_2h`, historical total-ratio `profit_over_baseline`, delays `12s / 24s / 36s`:

| Chain | Role | LSTM | Transformer | Transformer-LSTM |
| --- | --- | --- | --- | --- |
| Ethereum | Unsafe reference | `0.0243 / 0.0260 / 0.0257` | `0.0238 / 0.0251 / 0.0248` | `0.0244 / 0.0263 / 0.0262` |
| Ethereum | `safe_best` role | `0.0110 / 0.0124 / 0.0124` | `0.0105 / 0.0114 / 0.0115` | `0.0106 / 0.0121 / 0.0119` |
| Polygon | Unsafe reference | `0.0029 / 0.0043 / 0.0051` | `0.0019 / 0.0022 / 0.0026` | `0.0020 / 0.0030 / 0.0035` |
| Polygon | `safe_best` role | `0.0019 / 0.0037 / 0.0043` | `0.0018 / 0.0037 / 0.0044` | `0.0014 / 0.0029 / 0.0037` |
| Avalanche | Unsafe reference | `0.0125 / 0.0126 / 0.0135` | `0.0136 / 0.0156 / 0.0160` | `0.0116 / 0.0118 / 0.0117` |
| Avalanche | `safe_best` role | `0.0126 / 0.0127 / 0.0130` | `0.0083 / 0.0070 / 0.0074` | `0.0025 / 0.0009 / 0.0005` |

Checkpoint-selection parity, unsafe reference, `36s`, Polygon and Avalanche:

| Chain | Family | Economic Epoch | Validation-Loss Epoch | Economic Result | Validation-Loss Result |
| --- | --- | ---: | ---: | ---: | ---: |
| Polygon | LSTM | `2` | `3` | `0.0051` | `0.0048` |
| Polygon | Transformer | `3` | `21` | `0.0026` | `0.0036` |
| Polygon | Transformer-LSTM | `3` | `13` | `0.0035` | `0.0034` |
| Avalanche | LSTM | `9` | `1` | `0.0135` | `0.0151` |
| Avalanche | Transformer | `9` | `9` | `0.0160` | `0.0153` |
| Avalanche | Transformer-LSTM | `5` | `9` | `0.0117` | `0.0148` |

Checkpoint selection materially changes selected epochs in 5 of 6 cells. It does not uniformly reduce results toward paper bars, so the comparability caveat remains real but does not explain all Polygon/Avalanche above-paper behavior.

Distilled historical conclusions:

- The unsafe same-block reference remains the strongest professor-like comparator.
- `safe_best` improved the old safe baseline on all 9 chain-family cells in the cross-chain confirmation wave.
- `safe_best` is not a uniform paper-beating replacement for the unsafe reference.
- Ethereum safe-path HPO evals completed so far do not beat their `safe_best` baselines.
- Avalanche Transformer HPO improves over the weak safe baseline, but the wave remains incomplete until Avalanche Transformer-LSTM finishes.

### Open Benchmark Decisions

- Wait for jobs `57549`, `57550`, and `57551` before judging the targeted HPO wave.
- After the current HPO wave completes, decide whether tuned artifacts change the `safe_best` conclusion or only confirm that larger capacity/search is needed.
- Keep the unsafe same-block reference frozen as the professor-like comparator until the experimental surface is explicitly redefined.
- Do not promote `safe_best` to default architecture without an explicit decision.

### Planned Benchmark Sweeps

Large-capacity HPO remains planned after the active remote HPO wave finishes and local/remote configs are reconciled. Purpose: test whether remaining safe-path gaps are capacity or optimization limits rather than temporal-surface or feature-contract limits.

Target cells:

- Ethereum `transformer`.
- Ethereum `transformer_lstm`.
- Avalanche `transformer_lstm`.
- Avalanche `transformer`, if the current tuned result still needs deeper validation.

Trial budget:

- Start with `120` trials per cell.
- Review at `40` and `80` trials before spending the full budget.
- Continue past `120` only if recent trials improve best objective by at least `0.001` absolute `profit_over_baseline` or reveal a clearly new high-performing region.

Training search:

- `learning_rate: [0.00003, 0.0001, 0.0003]`.
- `weight_decay: [0.0, 0.0001, 0.001, 0.01]`.
- `batch_size: [64, 128, 256, 512]`.
- Keep high Slurm `ulimit -n`; do not force `SPICE_DATALOADER_WORKERS=0` unless open-file failures recur.

Transformer search:

- `d_model: [384, 512, 768, 1024, 1536]`.
- `transformer_layers: [4, 6, 8, 12]`.
- `nhead: [4, 8, 16]`, subject to `d_model % nhead == 0`.
- `feedforward_multiplier: [2, 4]`, resolved to concrete `feedforward_dim`.
- `head_hidden_dim: [256, 512, 1024]`.
- `dropout: [0.0, 0.1, 0.2, 0.3]`.

Transformer-LSTM search:

- `d_model: [384, 512, 768, 1024]`.
- `hidden_size: [384, 512, 768, 1024, 1536]`.
- `transformer_layers: [4, 6, 8]`.
- LSTM `num_layers: [1, 2, 3]`.
- `nhead: [4, 8, 16]`, subject to `d_model % nhead == 0`.
- `feedforward_multiplier: [2, 4]`, resolved to concrete `feedforward_dim`.
- `head_hidden_dim: [256, 512, 1024]`.
- `dropout: [0.0, 0.1, 0.2, 0.3]`.

Optional LSTM expansion, only if LSTM remains in scope:

- `hidden_size: [256, 384, 512, 768, 1024]`.
- `num_layers: [1, 2, 3, 4]`.
- `input_projection_dim: [128, 256, 512]`.
- `head_hidden_dim: [256, 512, 1024]`.
- `dropout: [0.0, 0.1, 0.2, 0.3]`.

Large-capacity tuning-space support is implemented for model-capacity fields. Benchmark command YAMLs are intentionally deferred until remote wave results and final launch cells/study names are settled.

Other planned sweeps:

- Re-run lookback-window sweeps for modern `same_block_closed` and `block_open_lagged`, including `900s` and longer windows. Earlier lookback evidence was mostly on estimated-block paths.
- Compare `current_row_nominal_window` against `current_row_recent_delta_window` using one simple median-of-training-deltas policy.
- Sweep sample counts such as `400k`, `1m`, and `3m` to separate data-volume gains from regime-drift costs.
- Sweep fixed `estimated_block` against modern current-row timestamp paths, keeping its paper-style nominal block grid while mapping offset `0` to the current row.

### Benchmarking Rules

- Treat `paper_replay_2h` results as historical remote evidence until rerun under current local evaluator names.
- Treat `poisson_replay_2h_mean` as the current canonical evaluator for new local benchmark claims.
- Keep `poisson_replay_2h_total` for total-ratio diagnostics and legacy comparability.
- Do not read notebook rollout/fullset diagnostics as equivalent to one-shot replay.
- Do not claim exact professor-pipeline parity; preprocessing, split construction, evaluator semantics, and checkpoint selection remain partially unresolved.

## Feature And Architecture Progress

### Current Surfaces

`same_block_closed` is the frozen unsafe same-block reference. It uses the current-block action space, fixed ex-ante classes, current-row pricing, and post-block features. It is unsafe because the model can act on the same block row whose finalized block facts it already sees.

`block_open_lagged` is the safe current-block surface. It keeps current base fee available but lags finalized current-block facts. It is the clean causal sibling of the unsafe reference.

`safe_best` is not a surface. It is a historical benchmark role combining per-family safe block-open feature and interval choices:

- LSTM: block-open, no broad time features, `recent_deltas`.
- Transformer: block-open, no `time_since_start`, nominal interval.
- Transformer-LSTM: block-open, calendar-only time, `recent_deltas`.

### Current Evaluators

`poisson_replay_2h_mean` is the primary evaluator for current work. `poisson_replay_2h_total` preserves total-ratio diagnostics. `zero_stop_rollout_fullset` and `anchor_basefee_fullset` remain diagnostic fullset evaluators.

Replay is a one-shot decoded-offset benchmark: the model commits to one decoded choice from the current row. Notebook-style rollout is a sequential re-decision policy and is easier to do well on, so it is diagnostic only.

### Feature Work

Completed feature findings kept for current relevance:

- Full time features were not the best safe-path choice for any family in the completed Ethereum ablation.
- Dropping `time_since_start` was neutral-to-helpful often enough to justify the family-specific `safe_best` role.
- `recent_deltas` helped some family/surface combinations but was not universal.
- Do not collapse safe-path feature choices to one uniform feature set only to reduce config count.

### Architecture Cleanup

Completed cleanup:

- Compiler runtime metadata serializes through compiler-id dispatch.
- Builder metadata remains typed, with compiler payloads rehydrated through the compiler registry.
- Workflow request models are task-specific.
- Remote execution submits resolved workflow config snapshots.
- Corpus, study, and artifact identity use resolved paths consistently, with dataset evaluation date included in corpus identity.

Remaining cleanup:

- Remove stale docs, dead codecs/configs, parity defaults, and redundant feature helpers after remote jobs finish or are archived.
- Keep raw `outputs/` artifacts and Slurm logs untracked.

## Research Direction

### Thesis / Internship Position

Internship 1 baseline replication is economically supported on this benchmark context: the unsafe professor-like reference reproduces or exceeds reported paper gains. This is not a perfect clone claim because professor preprocessing, split construction, model selection, and evaluator semantics are not fully identical.

Internship 1 optimization is partially supported by time-feature ablations, interval-estimator experiments, and the current HPO wave. Full Bayesian HPO, lookback sweeps, sample-count sweeps, and larger model-capacity searches remain planned work.

Thesis / Internship 2 direction remains aligned with uncertainty quantification and dynamic prediction-window sizing. Thesis-scale expansion should record completed current benchmark results in `benchmarks/results.csv`.

### Candidate Ideas

- Range and quantile position features: encode whether the current fee is near a recent local min, median, or high quantile rather than only exposing raw level and rolling means.
- Regime-spread and persistence features: encode short-vs-long pressure shifts, sustained rising/falling fee streaks, and whether congestion relief appears persistent.
- Cadence and opportunity-density features: encode recent inter-block timing, block-count density inside fixed time windows, and uncertainty in how many opportunities fit within the delay budget.
- Protocol-aware pressure features: encode EIP-1559 pressure directly through utilization relative to target and signed recent excess/slack.
- New data sources: consider mempool backlog, pending transaction age, priority-fee distributions, and block-composition features only if ingestion scope expands.

## Update Rules

- Every current-status claim needs a date, job id, artifact id, or config name.
- Do not use “latest” without timestamp and context.
- Replace active job status in place when jobs complete.
- Keep historical numeric tables only when they affect current decisions.
- Put planned benchmark work in `Planned Benchmark Sweeps`.
- Put speculative feature work in `Candidate Ideas`.
- Do not mix local current config names with remote old-branch preset names without stating which context applies.
- Keep completed experiment detail out of this file unless it changes an active decision; use `benchmarks/results.csv` for durable current-result provenance.

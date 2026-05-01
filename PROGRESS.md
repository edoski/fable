# Progress

_Last verified: 2026-04-28 08:12 CEST_

Current runnable defaults use evaluator `poisson_replay_2h` and objective `profit_poisson_replay_2h`. The sibling evaluator `full_temporal_replay` and objective `profit_full_temporal_replay` are available for deterministic all-supplied-sample temporal replay. Older evaluator ids in this file are historical remote-run evidence only.

## Status Snapshot

Local `main` is ahead of the remote experiment branch. That split is intentional: the just-completed remote wave ran on `codex/temporal-parity` with older config/evaluator names. Preserve the remote evidence before updating that checkout.

The current local architecture is safe-only. The runnable defaults are surface `current_row_fee_dynamics`, features `core_fee_dynamics`, problem `current_row_nominal`, compiler `observed_time_window`, execution policy `strict_deadline_miss`, and evaluator `poisson_replay_2h`.

Remote benchmark logs still use older historical preset and evaluator names such as `same_block_closed`, `block_open_lagged`, `estimated_block`, `icdcs_2026_oracle_intermediate`, `icdcs_2026_professor_block_open_*`, and `paper_replay_2h`. Those names are historical evidence only in the current codebase. The removed runnable paths are documented in `ARCHIVE.md`.

`safe_best` is a historical benchmark role, not a local runnable config. It means the best family-specific safe block-open choices found before the current cleanup: LSTM with no broad time features, Transformer without `time_since_start`, and Transformer-LSTM with calendar-only time plus `recent_median`.

As of this verification, the delay-sensitivity sweep, checkpoint-selection parity, and targeted `safe_best` HPO wave are complete. The local code has moved to the current safe-only architecture, and the default `current_row_nominal` problem now uses `1,000,000` samples.

Current corpus status:

- Ethereum `current_row_nominal`: local 1M corpus `cor_9a73b1e88edb488afb1e` validated on 2026-04-28. History has `1,093,256` contiguous rows, evaluation has `7,162` contiguous rows, enriched fee-history columns are finite, `core_fee_dynamics` builds a finite `(1,093,256, 22)` float32 matrix, and the observed-time-window store has `1,093,103` valid samples against the `1,000,000` requirement. It was acquired before the temporary 1M spec was folded into the default.
- Avalanche `current_row_nominal`: local 1M acquisition running on 2026-04-28 via provider `tenderly`, because PublicNode-style Avalanche endpoints did not support sufficiently deep historical `eth_feeHistory`. The process was launched before the temporary 1M spec was folded into the default.
- Polygon `current_row_nominal`: local 1M acquisition running on 2026-04-28 via provider `publicnode` with provider-owned conservative runtime limits. The process was launched before the temporary 1M spec was folded into the default.
- After ETH/AVAX/POL 1M corpora are validated locally, push corpora to remote, run a small remote smoke train/evaluate, then launch the current safe baseline replication grid.

## Benchmarking

### Current Benchmark Context

- Runnable surface: `current_row_fee_dynamics`.
- Runnable features: `core_fee_dynamics`.
- Default problem: `current_row_nominal` with `sample_count: 1,000,000`.
- Slot-spacing comparison problem: `current_row_recent_median`.
- Compiler: `observed_time_window`.
- Execution policy: `strict_deadline_miss`.
- Default evaluator: `poisson_replay_2h`, reporting event-mean `profit_over_baseline` and `cost_over_optimum`.
- Sibling evaluator: `full_temporal_replay`, scoring every supplied sample once through the same Temporal Accounting metrics.

Historical remote results below use removed runnable paths such as `same_block_closed`, `block_open_lagged`, and `estimated_block`. They remain useful as thesis evidence, not as current config targets.

Historical remote results below use the older `paper_replay_2h` total-ratio style unless stated otherwise. Do not silently compare those numbers against current `poisson_replay_2h` output.

### Active Remote Runs

Remote host: `edoardo.galli3@giano.cs.unibo.it`, storage root `/scratch.hpc/edoardo.galli3/spice/outputs`, log root `/scratch.hpc/edoardo.galli3/slurm`.

No active `safe_best` HPO jobs remain.

Completed targeted HPO chains:

| Cell | HPO Result | Tuned Eval | Notes |
| --- | --- | --- | --- |
| Ethereum LSTM | best `0.0140` | `0.0112` | materialized below HPO best and below safe pre-HPO `0.0124` |
| Ethereum Transformer | best `0.0137` | `0.0112` | materialized below HPO best and near safe pre-HPO `0.0115` |
| Ethereum Transformer-LSTM | best `0.0141` | `0.0115` | materialized below HPO best and near safe pre-HPO `0.0119` |
| Avalanche Transformer | best `0.0224` | `0.0122` | materialized below HPO best, above safe pre-HPO `0.0074`, below unsafe `0.0160` |
| Avalanche Transformer-LSTM | best `0.0236` | `0.0133` | materialized below HPO best, above safe pre-HPO `0.0005`, above same-family unsafe `0.0117` |

The HPO values are tuning/validation objective values. The tuned eval values are held-out `paper_replay_2h` results at `36s`.

Avalanche Transformer-LSTM durable run references: tune job `57549`, train job `57550`, eval job `57551`; tune best trial `17`; tune best value `0.0236`; tune best params `training.learning_rate=0.0001`, `training.weight_decay=0.05`, `model.hidden_size=128`, `model.d_model=256`, `model.dropout=0.1`; train artifact `/scratch.hpc/edoardo.galli3/spice/outputs/artifacts/avalanche/art_b994d827efd0878aa4d8`; train selected epoch `13`; train objective `0.0219`; evaluation id `paper_replay_2h-36s-124a240ff5442623`; eval events `17867`; final eval `profit_over_baseline=0.0133`.

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
- Ethereum safe-path HPO evals did not beat their `safe_best` baselines.
- Avalanche HPO improved the weak safe Transformer and Transformer-LSTM baselines after materialization, but both show a large HPO-to-final-eval gap.
- Avalanche Transformer-LSTM is the strongest materialized tuned safe cell in this wave and beats its same-family unsafe reference, but it still does not close the broader unsafe-reference gap across all Avalanche model families.

### Open Benchmark Decisions

- Preserve the completed remote HPO wave as historical `paper_replay_2h` total-ratio evidence; do not append it to `benchmarks/results.csv` unless rerun or re-evaluated under current evaluator semantics.
- Reconcile local and remote code before launching the next sweep.
- Preserve the unsafe same-block reference as archived professor-like evidence; do not keep it runnable in local architecture.
- Treat `safe_best` as a historical role. Do not promote it to default architecture without rerunning under current safe-only configs.

### Planned Benchmark Sweeps

First post-refactor ablation: `elapsed_seconds` / corpus-position signal.

Purpose: decide whether the old elapsed-time-style feature deserves to survive as an explicit experimental feature, or whether it should be removed from the current runnable feature implementation and docs. Historical evidence says full time features were not the best old safe-path choice and dropping `time_since_start` was neutral-to-helpful often enough to define `safe_best`, but that is not a current-semantics proof. The post-refactor ablation should answer the question under the clean safe architecture.

Required setup:

- Regenerate and validate ETH/AVAX/POL corpora with the expanded current RPC schema before running this ablation. Ethereum 1M is already locally validated as `cor_9a73b1e88edb488afb1e`; Avalanche and Polygon 1M acquisitions are running as of 2026-04-28.
- Train new artifacts from current configs only. Do not use old block-open, same-block, estimated-block, feature-family, or old-schema compatibility paths.
- Keep `core_fee_dynamics` as the baseline feature catalog.
- Use the explicit experimental features config `core_fee_dynamics_elapsed_position`, identical to `core_fee_dynamics` except for adding `elapsed_seconds`.
- Treat `elapsed_seconds` as a corpus-position feature: timestamp minus the first timestamp in the materialized feature table. It is not direct future leakage, but it can let models key on dataset position, long-term regime, or split-specific trends rather than reusable fee dynamics.

Minimal smoke pass:

- Run Polygon + LSTM first, same seed, same current surface/problem/evaluator, two variants: baseline `core_fee_dynamics` and experimental `core_fee_dynamics_elapsed_position`.
- Use the smoke pass only to decide whether a full cluster ablation is worth spending. Do not treat it as proof.

Proof-quality grid:

- Chains: Ethereum, Polygon, Avalanche.
- Models: LSTM, Transformer, Transformer-LSTM.
- Problem: `current_row_nominal`.
- Surface: `current_row_fee_dynamics`, with only the `features` selection varied.
- Historical primary evaluator id at the time: `poisson_replay_2h_mean`.
- Compare held-out evaluation results, not just training loss or HPO validation objective.
- Prefer repeated seeds if the single-seed deltas are small or inconsistent.

Decision rule:

- If `elapsed_seconds` does not materially and consistently improve held-out evaluation across chains/model families, remove it from the current runnable feature implementation and current docs. Preserve only the historical note that paper/professor-lineage code used an elapsed-time-style signal.
- If it helps materially and consistently, keep it out of the default until there is an explicit architecture decision about whether corpus-position signals are acceptable. It should remain an experimental features config, not part of `core_fee_dynamics`.

`large_capacity_hpo` is the planned calibration HPO benchmark after ETH/AVAX/POL 1M corpora are validated locally and pushed to remote. Purpose: calibrate each chain/model cell once, under the current safe surface and canonical mean-Poisson evaluator, without turning every structural sweep into an HPO sweep.

Target cells:

- Chains: Ethereum, Polygon, Avalanche.
- Models: LSTM, Transformer, Transformer-LSTM.
- Features/problem: `core_fee_dynamics`, `current_row_nominal`.

Trial budget:

- `40` trials per chain/model cell.
- Do not expand trial count until the 3x3 calibration results are collected and compared against the untuned baseline grid.
- Do not rerun HPO inside `lookback_window_sweep`, `slot_spacing_sweep`, `elapsed_position_ablation`, or `delay_degradation_sweep`.

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

LSTM search:

- `hidden_size: [256, 384, 512, 768, 1024]`.
- `num_layers: [1, 2, 3, 4]`.
- `input_projection_dim: [128, 256, 512]`.
- `head_hidden_dim: [256, 512, 1024]`.
- `dropout: [0.0, 0.1, 0.2, 0.3]`.

Launch decisions are now governed by the feature-set stabilization sequence below. Feature-set changes define the benchmark surface, so broad structural sweeps should wait until the input feature set is stable.

Feature-set stabilization sequence:

1. The submitted local-trends A/B grid completed cleanly. Local trends won 6/9 cells, tied Avalanche Transformer-LSTM, narrowly lost Polygon LSTM, and lost Avalanche Transformer.
2. The restored safe local-trend outputs are promoted into canonical `core_fee_dynamics`. This means `core_fee_dynamics` is now the expanded safe fee-dynamics set; do not keep a long-term separate local-trends feature axis.
3. Rerun `safe_baseline_grid` and `large_capacity_hpo` once on the promoted `core_fee_dynamics` definition. These runs establish the first clean post-promotion baseline and tuned reference.
4. Run `priority_fee_ablation` if priority fees need a current-semantics comparison. It compares canonical no-priority `core_fee_dynamics` against `core_fee_dynamics_with_priority_fee`, which adds lagged/public priority-fee scalars and p50/spread local trends.
5. Run `unsafe_core_fee_dynamics_ablation` to isolate same-block gas/tx leakage without priority-fee features in either arm.
6. Only after feature-set stabilization should the broader structural sweeps run: `slot_spacing_sweep`, `lookback_window_sweep`, `delay_degradation_sweep`, and optionally `elapsed_position_ablation`.

Priority-fee local-trends rationale:

- Base fee moves mechanically; priority-fee percentiles and spread can capture the urgency layer of the fee market before base fee fully reacts.
- MEV/private orderflow is not a leakage problem if the selected inputs remain lagged public `eth_feeHistory` summaries, but it can make priority-fee signals incomplete or chain-dependent because bundles may pay through private routes or coinbase transfers.
- Same-row priority-fee statistics remain forbidden because they are finalized current-block facts. The unsafe A/B is limited to gas/tx and fee-history gas-ratio leakage.

Benchmark sweep operator flow:

- Before launching a full sweep, push validated ETH/AVAX/POL 1M corpora to remote and run one small current-config smoke train/evaluate to verify the remote checkout, storage, and evaluator path.
- Create the exact remote workflow DAG with `spice benchmark plan <name> --target disi_l40`. This writes `outputs/benchmarks/runs/<name>/<timestamp>/metadata.json` and `plan.jsonl`.
- Submit with `spice benchmark submit outputs/benchmarks/runs/<name>/<timestamp>`. This records the remote git commit and Slurm submission records in `submission.jsonl`.
- Collect only after the sweep has finished: `spice benchmark collect outputs/benchmarks/runs/<name>/<timestamp>`. Collection is all-or-nothing and requires each persisted evaluation summary to match the submitted `execution_ref`.
- Export human-readable results with `spice benchmark index export --output benchmarks/results.csv`. Rebuild/query thesis result state with `spice benchmark index rebuild` and `spice benchmark index list`.

Deferred architecture roadmap after the current cleanup sweep:

- Temporal Replay Runner generalization only when a second decoded-result Adapter exists.
- Fine-grained post-window eligibility policy if we deliberately choose to change sample eligibility/performance.

Configured sweep specs awaiting launch decisions:

- `safe_baseline_grid`: untuned ETH/POL/AVAX by LSTM/Transformer/Transformer-LSTM on `current_row_fee_dynamics`, `core_fee_dynamics`, default 1M `current_row_nominal`, and `poisson_replay_2h`.
- `large_capacity_hpo`: bounded calibration HPO over the same 3x3 chain/model grid, large-capacity tuning spaces, 40 trials per cell, tuned train, and tuned `poisson_replay_2h` evaluation.
- `priority_fee_ablation`: fixed train/evaluate comparison of canonical `core_fee_dynamics` against `core_fee_dynamics_with_priority_fee` across the same 3x3 safe grid. No per-cell HPO.
- `unsafe_core_fee_dynamics_ablation`: fixed train/evaluate comparison of canonical `core_fee_dynamics` against `core_fee_dynamics_unsafe` across the same 3x3 safe grid. No per-cell HPO.
- `lookback_window_sweep`: fixed train/evaluate grid over ETH/POL/AVAX, LSTM/Transformer/Transformer-LSTM, and `600s`/`900s`/`1200s` lookbacks. No per-cell HPO.
- `slot_spacing_sweep`: fixed train/evaluate comparison of `current_row_nominal` and `current_row_recent_median` across the same 3x3 safe grid. No per-cell HPO.
- `elapsed_position_ablation`: fixed train/evaluate comparison of `core_fee_dynamics` against `core_fee_dynamics_elapsed_position` across the same 3x3 safe grid.
- `delay_degradation_sweep`: fixed train/evaluate ladder over `max_delay_seconds = 12, 24, 36, 48, 60, 72, 84, 96, 108, 120`. It trains one artifact per delay and evaluates with the same delay via the default `evaluation.delay_seconds = problem.max_delay_seconds` rule.

GPU-hour policy:

- Do not tune every structural sweep cell. Run `large_capacity_hpo` as the broad calibration search for the current canonical feature set, rerun it only after a material feature promotion, then run structural sweeps as fixed train/evaluate grids.
- If tuned parameters should be reused outside their exact study identity, first materialize explicit model/training presets from the selected HPO results. Current benchmark YAML does not automatically transplant best parameters across different problem/features cells.
- Keep `poisson_replay_2h` as the default benchmark evaluator. Use `evaluator_objective_grid` when comparing Poisson replay against full temporal replay.

Metrics plan:

- Domain replay metrics stay in evaluators and benchmark collection. `poisson_replay_2h` is the default domain result; `full_temporal_replay` is the deterministic sibling result.
- Pre-benchmark v1 ML metrics: add `macro_f1` for both offset-output prediction families and `log_fee_mae` / `log_fee_mse` for `min_block_fee_multitask`, then persist them through `benchmarks/results.csv`. Do not do a broader metric redesign before this benchmark wave.

### Benchmarking Rules

- Treat `paper_replay_2h` results as historical remote evidence until rerun under current local evaluator names.
- Treat `poisson_replay_2h` as the current default evaluator for new local benchmark claims unless the claim is explicitly about full temporal replay or evaluator/objective comparison.
- Do not read notebook rollout/fullset diagnostics as equivalent to one-shot replay.
- Do not claim exact professor-pipeline parity; preprocessing, split construction, evaluator semantics, and checkpoint selection remain partially unresolved.

## Feature And Architecture Progress

### Current Safe Architecture

`current_row_fee_dynamics` is the primary current runnable surface. It composes `core_fee_dynamics`, 1M-sample `current_row_nominal`, `fixed_sequence_temporal`, `lstm`, `icdcs_2026`, `profit_poisson_replay_2h`, and `poisson_replay_2h` by default. Sample-count sweeps are deferred until exact chain/date ranges and protocol-regime boundaries are explicit.

`core_fee_dynamics` is safe by construction. The source layer allows current `base_fee_per_gas[t]` because EIP-1559 base fee for block `t` is deterministic from parent state before block `t` execution. Finalized current-block facts such as gas used, gas limit, transaction count, and fee-history gas-used ratio are lagged to `t-1`.

The feature matrix invariant is finite `float32` only. Pre-warmup placeholders may exist only to keep arrays aligned; invalid pre-warmup anchors are excluded before splitting. Required selected sources must be finite after warmup or matrix construction fails.

### Archived Historical Surfaces

`same_block_closed` is the frozen unsafe same-block reference. It uses the current-block action space, fixed ex-ante classes, current-row pricing, and post-block features. It is unsafe because the model can act on the same block row whose finalized block facts it already sees.

`block_open_lagged` is the safe current-block surface. It keeps current base fee available but lags finalized current-block facts. It is the clean causal sibling of the unsafe reference.

Both names are historical evidence only after the safe refactor. Current runnable work uses `current_row_fee_dynamics`.

`safe_best` is not a surface. It is a historical benchmark role combining per-family safe block-open feature and interval choices:

- LSTM: block-open, no broad time features, `recent_median`.
- Transformer: block-open, no `time_since_start`, nominal interval.
- Transformer-LSTM: block-open, calendar-only time, `recent_median`.

### Current Evaluators

`poisson_replay_2h` is the default evaluator for current work. `full_temporal_replay` is the objective-capable sibling evaluator that scores every supplied sample once. Removed total-ratio and older fullset evaluator ids in historical notes are not runnable current configs.

Replay is a one-shot decoded-offset benchmark: the model commits to one decoded choice from the current row. Notebook-style rollout is a sequential re-decision policy and is easier to do well on, so it is diagnostic only.

### Feature Work

Completed historical feature findings kept for current relevance:

- Full time features were not the best old safe-path choice for any family in the completed Ethereum ablation.
- Dropping `time_since_start` was neutral-to-helpful often enough to justify recording the historical `safe_best` role.
- The old recent-delta interval estimate helped some family/surface combinations but was not universal.

Current runnable feature work intentionally uses one canonical safe catalog, `core_fee_dynamics`. New feature ideas should enter as explicit catalog outputs or future catalogs only after they have a clear source policy, warmup/null policy, and benchmark reason.

### Architecture Cleanup

- Remove stale docs, dead codecs/configs, parity defaults, and redundant feature helpers after remote jobs finish or are archived.
- Keep raw `outputs/` artifacts and Slurm logs untracked.

## Research Direction

### Thesis / Internship Position

Internship 1 baseline replication is economically supported on this benchmark context: the unsafe professor-like reference reproduces or exceeds reported paper gains. This is not a perfect clone claim because professor preprocessing, split construction, model selection, and evaluator semantics are not fully identical.

Internship 1 optimization is partially supported by time-feature ablations, interval-estimator experiments, and the current HPO wave. Lookback sweeps, sample-count sweeps, slot-spacing sweeps, and larger model-capacity searches remain to run.

Thesis / Internship 2 direction remains aligned with uncertainty quantification and dynamic prediction-window sizing. Thesis-scale expansion should record completed current benchmark results in `benchmarks/results.csv`.

### Candidate Ideas

- Binance market data: defer spot/futures market-source ingestion. Candidate features include ETH/AVAX/POL spot returns over short windows, BTC market return, token-vs-BTC relative return, realized volatility, traded volume, taker imbalance, futures premium, and funding-rate state. Compute these on market time first, then as-of join by `available_at`, not raw market timestamp. For klines, `available_at` must be close time plus publication lag because open time alone leaks close/high/low/volume.
- Receipts/log aggregates: defer receipt/log-derived activity features. Candidate outputs include failed transaction ratio, effective gas price percentiles, receipt gas-used summaries, log count, contract-call density, and simple ERC20/DEX activity proxies. These need a separate ingestion cost decision because they require receipt/log pulls beyond block headers and `eth_feeHistory`.
- Producer/author metadata: defer proposer/miner/author features. Potential value is validator/proposer behavior or builder/producer-specific fee dynamics, but cross-chain semantics are unclear and raw identifiers risk memorization. Any future version should require a source policy, hashing/grouping policy, minimum support threshold, and explicit ablation.
- Blob and block-size experiments: keep canonical nullable fields for `block_size_bytes`, `blob_gas_used`, and `excess_blob_gas`, but do not select these features by default. Future experiments should treat support as chain/date dependent and require finite selected values after warmup.
- Slot-spacing sweep: compare `current_row_nominal` and `current_row_recent_median` after current jobs are archived/synced. `recent_median` is scoped only to `observed_time_window.slot_spacing`; do not reuse that name for feature concepts.
- Sample-count sweep: deferred on 2026-04-28. The current default is 1M samples. Future 3M or larger comparisons must first verify exact chain/date ranges and avoid crossing major fee/protocol regime transitions such as Ethereum Pectra/Fusaka or comparable Polygon/Avalanche fee-rule changes. Reintroduce the runnable benchmark only after the date-range policy is explicit.
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

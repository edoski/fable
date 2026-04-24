# Progress

## Current Benchmark Context

- Current unsafe surface: `same_block_closed`
- Current safe surface: `block_open_lagged`
- Current paper-style nominal-grid compiler: `estimated_block`
- Primary evaluator: `poisson_replay_2h`
- Diagnostic evaluators:
  - `zero_stop_rollout_fullset`
  - `anchor_basefee_fullset`
- Latest completed wave:
  - cross-chain confirmation of frozen same-block unsafe reference vs improved block-open safe candidate (`safe_best` historical role)
  - chains: Ethereum, Polygon, Avalanche
  - delay: `36s`

## Future Benchmark Sweeps

- Sweep fixed `estimated_block` against the modern current-row timestamp paths. It now keeps the paper-style nominal block grid while mapping offset `0` to the current row.
- Re-run lookback-window sweeps for modern `same_block_closed` and `block_open_lagged`, including `900s` and longer windows. Earlier lookback work was mostly on estimated-block paths and should not be treated as modern-path evidence.
- First wave:
  - surfaces: `same_block_closed`, `block_open_lagged`
  - lookback: `600s`, `900s`, `1200s`; consider `1800s` only after the smaller sweep is interpretable
  - feature sets: `full`, `no_time_since_start`, `calendar_only_time`
  - models: `lstm`, `transformer`, `transformer_lstm`
  - delays: `12s`, `24s`, `36s`
- Second wave:
  - compare `current_row_nominal_window` vs `current_row_recent_delta_window`
  - sweep recent-delta estimator window sizes such as `64`, `128`, `256`, `512` blocks
  - compare `median`, `mean`, and possibly quantile interval statistics if the estimator path remains promising
  - sweep `sample_count` such as `400k`, `1m`, `3m` to separate data-volume gains from regime-drift costs
- Third wave:
  - treat model/training capacity as HPO rather than benchmark semantics
  - existing knobs: learning rate, weight decay, batch size, hidden size, layer count, dropout, `d_model`, transformer layers
  - likely useful new tuning knobs: LSTM `input_projection_dim`, prediction-head `head_hidden_dim`, transformer `feedforward_dim`, and possibly `nhead`
  - defer `max_epochs` and early-stopping patience unless training curves show undertraining or premature stopping

## Overnight Checkpoint-Parity Run

- Code milestone:
  - generic `--objective` override landed for `train`, `tune`, and `evaluate`
  - `acquire` rejects objective overrides
  - targeted pytest, full `pytest -q`, `ruff check`, and `pyright` passed before remote launch
- Remote matrix:
  - queued on `disi_l40`
  - checkpoint parity: unsafe reference, Polygon/Avalanche, all 3 families, `36s`
  - delay sensitivity: unsafe reference and `safe_best` benchmark role, Ethereum/Polygon/Avalanche, all 3 families, `12s/24s/36s`
  - targeted `safe_best` benchmark role HPO: Avalanche Transformer-LSTM/Transformer; Ethereum Transformer-LSTM/Transformer/LSTM
- HPO recovery:
  - Avalanche `safe_best` benchmark role Transformer tune job `57391` failed after 4 trials with a transient pin-memory / open-file runtime error
  - stale dependents `57392` and `57393` were cancelled
  - replacement tune/train/eval chain `57403` / `57404` / `57405` was queued to resume the same study to 40 total trials
  - Ethereum `safe_best` benchmark role Transformer tune job `57397` failed after 5 trials with the same runtime class
  - because the repeated failure points at the Slurm file-descriptor limit, remaining open HPO chains were requeued with `ulimit -n 4096` in the batch wrapper
  - old open HPO jobs `57388`-`57390`, `57394`-`57405` were cancelled or superseded
  - replacement HPO chains:
    - Avalanche Transformer-LSTM: `57406` / `57407` / `57408`
    - Avalanche Transformer: `57409` / `57410` / `57411`
    - Ethereum Transformer-LSTM: `57412` / `57413` / `57414`
    - Ethereum Transformer: `57415` / `57416` / `57417`
    - Ethereum LSTM: `57418` / `57419` / `57420`

### Delay-Sensitivity Sweep

Decimal values below are `profit_over_baseline` on `poisson_replay_2h`.

| Chain | Role | LSTM 12/24/36 | Transformer 12/24/36 | Transformer-LSTM 12/24/36 |
| --- | --- | --- | --- | --- |
| Ethereum | Unsafe reference | `0.0243 / 0.0260 / 0.0257` | `0.0238 / 0.0251 / 0.0248` | `0.0244 / 0.0263 / 0.0262` |
| Ethereum | `safe_best` role | `0.0110 / 0.0124 / 0.0124` | `0.0105 / 0.0114 / 0.0115` | `0.0106 / 0.0121 / 0.0119` |
| Polygon | Unsafe reference | `0.0029 / 0.0043 / 0.0051` | `0.0019 / 0.0022 / 0.0026` | `0.0020 / 0.0030 / 0.0035` |
| Polygon | `safe_best` role | `0.0019 / 0.0037 / 0.0043` | `0.0018 / 0.0037 / 0.0044` | `0.0014 / 0.0029 / 0.0037` |
| Avalanche | Unsafe reference | `0.0125 / 0.0126 / 0.0135` | `0.0136 / 0.0156 / 0.0160` | `0.0116 / 0.0118 / 0.0117` |
| Avalanche | `safe_best` role | `0.0126 / 0.0127 / 0.0130` | `0.0083 / 0.0070 / 0.0074` | `0.0025 / 0.0009 / 0.0005` |

### Checkpoint-Selection Parity

Scope: unsafe reference only, `poisson_replay_2h`, `36s`, Polygon and Avalanche.

| Chain | Family | Economic epoch | Validation-loss epoch | Same epoch? | Economic result | Validation-loss result |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Polygon | LSTM | `2` | `3` | No | `0.0051` | `0.0048` |
| Polygon | Transformer | `3` | `21` | No | `0.0026` | `0.0036` |
| Polygon | Transformer-LSTM | `3` | `13` | No | `0.0035` | `0.0034` |
| Avalanche | LSTM | `9` | `1` | No | `0.0135` | `0.0151` |
| Avalanche | Transformer | `9` | `9` | Yes | `0.0160` | `0.0153` |
| Avalanche | Transformer-LSTM | `5` | `9` | No | `0.0117` | `0.0148` |

Checkpoint parity read:

- Validation-loss selection materially changes the selected epoch in 5 of 6 cells.
- It does not uniformly reduce results toward the paper bars.
- Polygon LSTM and Transformer-LSTM are nearly unchanged; Polygon Transformer improves under validation loss.
- Avalanche LSTM and Transformer-LSTM improve under validation loss; Avalanche Transformer selects the same epoch and is close.
- The checkpoint-selection caveat remains real, but it does not by itself explain all Polygon/Avalanche above-paper behavior.

## Main Paths

- Unsafe reference path:
  - surface: `same_block_closed`
  - problem family: `current_row_nominal_window*`
  - feature sets: `same_block_closed_full*`
  - semantics: current-block action space, fixed ex-ante classes, current-row pricing, post-block features
  - why unsafe: model can act on the same block row whose finalized block facts it already sees
  - status: frozen unsafe same-block reference and main comparator
  - governance: do not reinterpret, optimize, or rename it until the current experimental surface is locked

- Safe current-block path:
  - surface: `block_open_lagged`
  - problem family: `current_row_nominal_window*`
  - feature sets: `block_open_lagged_full*`
  - semantics: current-block action space, fixed ex-ante classes, current-row pricing, block-open feature contract
  - safety rule: finalized current-block features are lagged; current base fee is kept
  - status: clean causal sibling path; current improved benchmark role is `safe_best`

- `safe_best` historical benchmark role:
  - `lstm`:
    - benchmark role: explicit `block_open_lagged` case
    - feature policy: `no_time_features`
    - interval policy: `recent_deltas`
  - `transformer`:
    - benchmark role: explicit `block_open_lagged` case
    - feature policy: `no_time_since_start`
    - interval policy: nominal
  - `transformer_lstm`:
    - benchmark role: explicit `block_open_lagged` case
    - feature policy: `calendar_only_time`
    - interval policy: `recent_deltas`
  - rationale:
    - time-feature ablations showed different winners by model family
    - interval estimation helped some families but not all
    - do not collapse to one uniform safe feature set merely to reduce YAML count

## Completed Cross-Chain Confirmation Wave

Decimal values below are `profit_over_baseline` on `poisson_replay_2h`.

| Chain | Paper Fig. 6 approx | Unsafe reference | Safe best |
| --- | --- | --- | --- |
| Ethereum | `0.0255 / 0.0262 / 0.0258` | `0.0248 / 0.0257 / 0.0255` | `0.0123 / 0.0113 / 0.0114` |
| Polygon | `0.0020 / 0.0020 / 0.0019` | `0.0043 / 0.0042 / 0.0028` | `0.0045 / 0.0039 / 0.0042` |
| Avalanche | `0.0095 / -0.0015 / 0.0058` | `0.0144 / 0.0133 / 0.0120` | `0.0132 / 0.0072 / 0.0031` |

Model order: `LSTM / Transformer / Transformer-LSTM`

Late completed diagnostic evals:

- Avalanche unsafe `transformer_lstm`:
  - `poisson_replay_2h`: `0.0120`
  - `zero_stop_rollout_fullset`: `0.0340`
  - `anchor_basefee_fullset`: `0.0102`
- Avalanche `safe_best` benchmark role `transformer_lstm` completed with:
  - `poisson_replay_2h`: `0.0031`
  - `zero_stop_rollout_fullset`: `0.0023`
  - `anchor_basefee_fullset`: `0.0031`

Old safe block-open baseline vs current `safe_best` benchmark role:

| Chain | Old safe block-open | Current `safe_best` role |
| --- | --- | --- |
| Ethereum | `0.0104 / 0.0112 / 0.0089` | `0.0123 / 0.0113 / 0.0114` |
| Polygon | `0.0040 / 0.0023 / 0.0036` | `0.0045 / 0.0039 / 0.0042` |
| Avalanche | `0.0121 / 0.0047 / 0.0009` | `0.0132 / 0.0072 / 0.0031` |

Final read:

- Ethereum:
  - unsafe remains near the paper band
  - `safe_best` improves the old safe baseline but remains below paper and unsafe
- Polygon:
  - `safe_best` generalizes well and beats both paper and the old safe baseline
  - unsafe also beats paper
- Avalanche:
  - unsafe beats paper across all 3 families
  - `safe_best` LSTM beats paper
  - `safe_best` Transformer is positive and above the paper Transformer bar, but below unsafe
  - `safe_best` Transformer-LSTM is positive but below the paper Transformer-LSTM bar
- Overall:
  - `safe_best` improves the old safe baseline on all 9 chain-family cells
  - `safe_best` generalizes beyond Ethereum in the sense that it improves the safe baseline on Polygon and Avalanche
  - `safe_best` is not a uniform paper-beating replacement for the unsafe reference
  - unsafe reference remains the strongest historical professor-like comparator and should stay frozen

No new wave has been queued.

## Verified Facts

- The professor artifacts are likely using an unsafe current-block convention.
- The professor-like path appears to use fixed ex-ante classes.
- The old realized-future action-mask oracle was too unsafe and is not the reference target.
- Unsafe intermediate is closer to the professor setup than the old realized-mask oracle.
- Notebook rollout is more permissive than replay because it re-decides one row at a time until the model emits `0`.

## Important Comparability Caveats

- Checkpoint selection differs from the professor reference code.
  - In SPICE, model selection / early-stopping priority is economic: `profit_over_baseline`.
  - In the professor training code, the visible selection rule appears to be validation loss.
  - This can materially change final economic results, especially on tighter-margin chains like Polygon.
- Our primary replay evaluator is stricter than the notebook-style evaluator.
  - `poisson_replay_2h` is a one-shot replay benchmark: the model commits to one decoded choice from the current row.
  - The professor notebook rollout is a sequential re-decision policy: move forward one row at a time until the model emits `0`.
  - So notebook rollout is easier to do well on and should not be read as directly equivalent to replay.
- “Better than paper” should be stated carefully.
  - What we can say confidently: on our benchmark context, our implementation reproduces or exceeds the reported economic gains.
  - What remains partially unresolved: exact parity with the professor's unpublished preprocessing, split construction, and model-selection pipeline.
- The unsafe reference is the best professor-like reproduction, but it is still not claimed to be a perfect clone.
  - It is the closest current match to the visible professor artifacts.
  - It remains useful as a reference path and upper-bound-like comparator, not as proof of exact experimental identity.

## Cross-Chain Replay Results

Percent values below are `profit_over_baseline` on `poisson_replay_2h`.

| Chain | Paper Fig. 6 approx | Unsafe reference | Safe block-open |
| --- | --- | --- | --- |
| Ethereum | `2.55 / 2.62 / 2.58` | `2.59 / 2.62 / 2.59` | `1.04 / 1.12 / 0.89` |
| Polygon | `0.20 / 0.20 / 0.19` | `0.46 / 0.34 / 0.32` | `0.40 / 0.23 / 0.36` |
| Avalanche | `0.95 / -0.15 / 0.58` | `1.25 / 1.35 / 1.23` | `1.21 / 0.47 / 0.09` |

Model order: `LSTM / Transformer / Transformer-LSTM`

## What Worked

- Unsafe current-block reference reproduces or exceeds the paper band across all three chains.
- Safe block-open remains viable but is materially below the unsafe reference on Ethereum.
- Cross-chain baseline picture is now complete for both safe and unsafe paths.

## What Did Not Work

- The fully safe next-block oracle path did not preserve the professor-like gain.
- Safe block-open still loses too much of the Ethereum gain.
- Split-local preprocessing is deferred; it is not the current architectural direction.

## Latest Completed Wave

- Wave: Ethereum-only time-feature ablations
- Scope:
  - both paths: unsafe reference and safe block-open
  - all 3 families
  - feature variants:
    - `full`
    - `no_time_since_start`
    - `no_time_features`
    - `calendar_only_time`
    - `time_since_start_only`
- Primary evaluator: `poisson_replay_2h`

### Unsafe reference replay results

| Family | Full | No `time_since_start` | No time features | Calendar only | `time_since_start` only |
| --- | --- | --- | --- | --- | --- |
| LSTM | `0.0228` | `0.0264` | `0.0266` | `0.0265` | `0.0255` |
| Transformer | `0.0265` | `0.0258` | `0.0261` | `0.0253` | `0.0263` |
| Transformer-LSTM | `0.0262` | `0.0264` | `0.0262` | `0.0261` | `0.0259` |

### Safe block-open replay results

| Family | Full | No `time_since_start` | No time features | Calendar only | `time_since_start` only |
| --- | --- | --- | --- | --- | --- |
| LSTM | `0.0096` | `0.0113` | `0.0119` | `0.0116` | `0.0096` |
| Transformer | `0.0099` | `0.0122` | `0.0106` | `0.0116` | `0.0110` |
| Transformer-LSTM | `0.0107` | `0.0121` | `0.0117` | `0.0123` | `0.0109` |

### Main conclusions

- `time_since_start` alone does **not** preserve most of the gain.
- Dropping `time_since_start` is often neutral-to-helpful and is a credible simplification candidate.
- On the safe path, full time features were never the best completed variant for any family.
- The strongest simplifying wins are:
  - safe `lstm`: `no_time_features` (`0.0119`) over full (`0.0096`)
  - safe `transformer`: `no_time_since_start` (`0.0122`) over full (`0.0099`)
  - safe `transformer_lstm`: `calendar_only_time` (`0.0123`) over full (`0.0107`)
- There is no single universal winner across all families.
- Notebook metrics were directionally consistent with the replay findings.

## Interval-Estimator Rationale

- This section records why the interval-estimation wave was run.
- The implementation exists and the first Ethereum matrix is completed below.
- Interval estimation:
  - Goal:
    - improve fixed ex-ante action sizing without using future realized timestamps
    - recover some of the unsafe-path gain while staying causal
  - Core requirement:
    - chain agnostic
    - online-safe
    - past-only
    - no launch until results from the current wave are reviewed in-thread
  - Why this matters:
    - the task is time-budgeted, not purely block-count-budgeted
    - a fixed nominal interval like `12s` or `1.6s` is simple but crude
    - if the interval is set too high, the action space is too narrow
    - if it is set too low, the action space is too wide and overflow / miss behavior increases
    - a better causal estimate may recover some performance without reintroducing leakage
  - Planned seam:
    - explicit `action_interval_estimator` inside the timestamp-window compiler
    - no chain-specific branching in acquisition or evaluation
    - artifact stores the resolved estimator provenance needed for rebuilds and auditability
  - Candidate estimator families:
    - `nominal`
      - use `chain.runtime.nominal_block_time_seconds`
      - baseline and default comparator
    - `recent_deltas_mean`
      - mean of recent positive inter-block deltas over the last `N` blocks
    - `recent_deltas_median`
      - median of recent positive inter-block deltas over the last `N` blocks
      - likely more robust than mean
    - `recent_deltas_quantile`
      - quantile of recent positive inter-block deltas, e.g. p25 / p50 / p75
      - gives a tunable conservative or aggressive width
    - possible later extensions if needed:
      - exponentially weighted versions
      - clipped / winsorized variants
      - chain-conditioned but still generic parameter sets
  - Likely first practical candidates:
    - rolling median over recent deltas
    - rolling quantile over recent deltas
    - these are simple, robust, and easy to explain
  - Semantics:
    - estimator resolves one fixed interval for the artifact from the selected training chronology
    - that interval determines the artifact's fixed ex-ante action width
    - evaluation must reuse the trained artifact width
    - evaluation must not inspect future realized evaluation-day timestamps to resize the action space
  - Safety rule:
    - estimator may use only past positive deltas available at training / deployment time
    - no realized future candidate-count help in loss, decode, or eval
  - Benchmark plan when this phase begins:
    - Ethereum `36s` first
    - both paths:
      - same-block unsafe reference
      - safe block-open
    - all 3 families
    - compare nominal vs recent-deltas estimators on the same evaluator
    - `poisson_replay_2h` remains primary
  - What would count as success:
    - clear replay improvement on safe block-open without architectural degradation
    - ideally preserves or improves unsafe reference too, but safe-path improvement is the main target
  - Failure modes to watch:
    - estimator overfits training chronology and does not transfer
    - estimator width is too optimistic and increases misses
    - estimator width is too conservative and collapses useful action options
    - gains are chain-specific and do not justify promotion
  - Governance:
    - no compiler default change without explicit in-thread approval

## Latest Completed Wave

- Wave: Ethereum-only interval-estimation matrix
- Scope:
  - safe reference
  - safe candidate
  - unsafe reference
  - all 3 families
  - estimators:
    - `nominal`
    - `recent_deltas`
- Primary evaluator: `poisson_replay_2h`

### Safe reference replay results

| Family | Nominal | `recent_deltas` |
| --- | --- | --- |
| LSTM | `0.0096` | `0.0115` |
| Transformer | `0.0108` | `0.0117` |
| Transformer-LSTM | `0.0115` | `0.0107` |

### Safe candidate replay results

| Family | Nominal | `recent_deltas` |
| --- | --- | --- |
| LSTM | `0.0124` | `0.0125` |
| Transformer | `0.0129` | `0.0120` |
| Transformer-LSTM | `0.0110` | `0.0116` |

### Unsafe reference replay results

| Family | Nominal | `recent_deltas` |
| --- | --- | --- |
| LSTM | `0.0254` | `0.0258` |
| Transformer | `0.0264` | `0.0259` |
| Transformer-LSTM | `0.0258` | `0.0254` |

### Main conclusions

- The per-family safe candidate path beats the safe reference path under both estimators for all 3 families.
- `recent_deltas` is helpful but not universal.
- On the safe reference path, `recent_deltas` helps `lstm` and `transformer`, but hurts `transformer_lstm`.
- On the safe candidate path, `recent_deltas` is:
  - neutral/slightly positive for `lstm`
  - negative for `transformer`
  - positive for `transformer_lstm`
- On the unsafe reference path, `recent_deltas` only moves results slightly and does not change the overall picture.
- Best safe replay results from this wave are:
  - `lstm`: safe candidate + `recent_deltas` = `0.0125`
  - `transformer`: safe candidate + nominal = `0.0129`
  - `transformer_lstm`: safe candidate + `recent_deltas` = `0.0116`

## Near-Term Decision Queue

- Decide whether the `safe_best` benchmark role becomes the working safe reference.
  - this is not automatic default promotion
  - keep the unsafe reference frozen as the historical professor-like comparator
  - keep per-family `safe_best` feature/interval choices unless results justify simplification
- Run a small checkpoint-selection parity check before major new feature work.
  - compare economic-objective selection against validation-loss selection
  - reason: this directly addresses why our Polygon/Avalanche results can exceed paper bars
  - goal: improve comparability claims without changing the feature contract
- Do not queue additional experiment waves automatically.
  - feature engineering, UQ, dynamic windows, and cleanup promotion all require explicit in-thread approval

## Thesis / Internship Position

- Internship 1 baseline-replication goal:
  - economically, the unsafe professor-like reference reproduces or exceeds the reported paper gains on the project benchmark context
  - exact professor-pipeline parity remains caveated because preprocessing, splits, evaluator semantics, and checkpoint selection are not fully identical
  - this is sufficient for a careful claim that the same research problem and model families have been reproduced on a comparable benchmark context, not for claiming a perfect clone
- Internship 1 optimization goal:
  - partially accomplished through time-feature ablations and interval-estimator experiments
  - full Bayesian HPO, loss-weight sweeps, and lookback-horizon sweeps remain optional future work rather than completed work
- Thesis / Internship 2 direction:
  - uncertainty quantification and dynamic prediction-window sizing remain aligned with the codebase direction
  - the cleanup plan supports these ideas by making evaluator compatibility, decoded prediction outputs, and temporal metadata ownership explicit
  - a committed benchmark ledger should be added before thesis-scale experiment expansion

## Deferred Future Feature Ideas

- Range-position features
  - Goal: tell the model where the current fee sits inside a recent local range, not just the raw fee level.
  - Candidate features:
    - distance to rolling min / rolling max
    - normalized position inside rolling min-max band
    - percentile rank of current fee within the last `N` blocks
    - drawdown from recent local maximum
  - Why this may help:
    - the decision problem is economic and wait-based
    - “am I already near a local cheap point?” is often more important than absolute level
    - we already expose rolling minima and means, but not the current fee's relative position inside the recent range
  - Safety notes:
    - straightforward to make causal on both unsafe and safe paths
    - for safe block-open, use current known base fee and past-only windows

- Regime spread features
  - Goal: make regime shifts explicit instead of forcing the model to infer them from separate short and long rolling stats.
  - Candidate features:
    - `roll10_mean_logfee - roll200_mean_logfee`
    - `roll10_std_logfee / roll200_std_logfee`
    - short-vs-long spreads for gas pressure / utilization
    - short-vs-long trend-slope differences
  - Why this may help:
    - these directly encode whether the chain is heating up, cooling off, or remaining stable
    - they are especially relevant for “wait one more block or not?” decisions
  - Safety notes:
    - cheap to add; fully causal if built from the existing rolling stats

- Curvature features
  - Goal: capture acceleration / deceleration, not just direction.
  - Candidate features:
    - second difference of log base fee
    - second difference of gas pressure
    - change in trend slope over recent windows
    - acceleration of utilization relative to target
  - Why this may help:
    - first differences say where the series is moving
    - curvature says whether that move is strengthening or weakening
    - this is directly relevant to timing decisions near turning points
  - Safety notes:
    - causal if derived from current-and-past safe observables only

- Persistence / streak features
  - Goal: encode how persistent the current congestion or relief regime has been.
  - Candidate features:
    - consecutive blocks with positive `dlog_base_fee`
    - consecutive blocks with negative `dlog_base_fee`
    - consecutive blocks with gas utilization above target
    - fraction of last `N` blocks above target
    - fraction of last `N` blocks with falling base fee
  - Why this may help:
    - rolling averages can hide whether the signal came from one spike or a sustained run
    - persistence is useful for judging whether mean reversion is likely
  - Safety notes:
    - for safe block-open, compute from lagged realized history and current known base fee only

- Protocol-aware pressure features
  - Goal: encode fee-relevant pressure in the protocol's own terms, not just generic utilization.
  - Candidate features:
    - `gas_utilization - target_utilization`
    - rolling mean of positive excess over target
    - rolling mean of slack below target
    - share of recent blocks materially above target
    - signed cumulative excess over the last `N` blocks
  - Why this may help:
    - EIP-1559 reacts to utilization relative to the target, not just raw gas usage
    - this is more directly tied to next-block fee dynamics
  - Safety notes:
    - works with current data; no new source required

- Cadence uncertainty / opportunity-density features
  - Goal: model not just fee movement, but how many timing opportunities likely fit inside the wait budget.
  - Candidate features:
    - rolling mean / std / coefficient of variation of `dt_seconds`
    - recent count of blocks in the last `60s`, `120s`, `300s`
    - fraction of recent blocks faster than nominal interval
    - recent opportunity density implied by observed inter-block times
  - Why this may help:
    - the action space is fundamentally time-budgeted
    - more blocks arriving quickly means more chances to wait without overshooting the deadline
  - Safety notes:
    - especially relevant for the future interval-estimation wave
    - must remain past-only on the safe path

- Robust quantile features
  - Goal: make the feature surface less sensitive to heavy tails and transient spikes.
  - Candidate features:
    - rolling fee quantiles: p10 / p25 / p50 / p75 / p90
    - rolling pressure quantiles
    - interquartile range
    - distance from current level to rolling median / p10 / p90
  - Why this may help:
    - mean / std are often brittle on fee series with bursts
    - quantiles can encode local distribution shape more robustly
  - Safety notes:
    - more expensive than mean/std, but still feasible with current data volumes

- New-data-source features if we expand ingestion later
  - Goal: move beyond post-block summaries into more directly causal inclusion-pressure signals.
  - Candidate sources:
    - mempool backlog / arrival-rate features
    - pending transaction age distribution
    - priority-fee / tip distribution
    - transaction-count / block-composition features
  - Why this may help:
    - these would likely be more powerful than extra calendar tricks if collected cleanly
    - they may allow a stronger same-slot or late-slot causal formulation
  - Constraint:
    - not part of the current data surface
    - requires explicit ingestion and problem-definition work

## Remaining Cleanup / Architecture Work

- Remaining architecture cleanup:
  - restore compiler ownership of temporal runtime metadata
  - make builder metadata typed and registry-owned
  - generalize prediction decoded outputs beyond `DecodedOffsets`
  - split large evaluator implementation into registry, engines, sampler policies, metrics, and summaries
  - make workflow request models task-specific
  - submit resolved workflow snapshots remotely instead of re-resolving request JSON on the cluster
  - clean corpus/study/artifact identity semantics
  - remove dead codecs, stale docs, parity defaults, and redundant feature helpers with justification

## Deferred Benchmark Ledger

- Keep SQLite artifact state as operational provenance and the query source for local/remote runs.
- Do not use SQLite as the committed canonical benchmark ledger.
  - reason: binary state is hard to diff, review, merge, and separate from partial local runs
- Keep `PROGRESS.md` as the narrative experiment log, not the structured source of truth.
- Add a committed benchmark ledger later under `benchmarks/`:
  - wave metadata: `benchmarks/waves/<wave_id>.yaml`
  - result rows: `benchmarks/results/<wave_id>.csv` or JSONL
  - fields: commit, branch, chains, models, surfaces, feature policies, evaluators, objectives, delay, metric, artifact ids, evaluation ids, job ids, row counts, final metric values
  - export command: read artifact SQLite state and write stable CSV/JSONL rows once a wave is ready to preserve
- Keep raw `outputs/` artifacts and Slurm logs untracked.

# FABLE

FABLE (Fee Analysis through Blockchain Learning and Estimation) is a closed-parent, fixed-block-horizon system for learning when a future block is likely to minimize base fee per gas. This manual is the canonical detailed account of the product's scientific contract, worked decision, architecture, interfaces, requests, durable objects, commands, operator configuration, serving surface, evaluation schemas, limitations, and sources.

FABLE derives from and extends selected temporal work from *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments*. FABLE is neither SPICE nor a reproduction of SPICE. Domain terms are defined in [CONTEXT.md](CONTEXT.md); durable-object and execution-boundary decisions remain recorded in [docs/adr/](docs/adr/).

## Contents

- [Overview](#overview)
- [One decision, end to end](#one-decision-end-to-end)
- [Scientific contract](#scientific-contract)
- [Architecture and deep interfaces](#architecture-and-deep-interfaces)
- [Exact reference](#exact-reference)
- [Limitations and sources](#limitations-and-sources)

## Overview

FABLE is organized around strict request values, direct owner functions, native library objects, and UUID-addressed durable objects. Dependencies point from operator edges toward scientific owners.

### System shape

```text
completed canonical Corpus pair
        |
        v
strict workflow request --> CLI or direct Python call
        |
        +--> tuning -----> Study
        +--> fitting ----> native Lightning artifact
        +--> evaluation -> observations.parquet
        |
        v
transient observation-derived reductions
```

`fable.config` owns frozen Pydantic values and small discriminated unions. `fable.requests` mints fresh UUIDv4 instances. A boundary receiving raw JSON or durable bytes hydrates the owning typed value once; downstream code trusts that value.

### Dependency direction

```text
CLI / serving / Python callers
                |
                v
execution, study, evaluation
                |
                v
modeling, temporal, min_block_fee, corpus
                |
                v
strict config values and canonical addresses
```

Each owner has one system seam:

- `corpus` owns canonical `BlockFrame` row truth and completed Corpus association.
- `temporal` owns causal feature state, fixed-block context/outcome geometry, and lazy historical examples.
- `min_block_fee` owns target state, the fixed training loss, two-head output, and decode.
- `modeling` owns the three concrete neural definitions, Lightning fitting, and native checkpoint loading.
- `study` owns bounded candidate membership, ordered retained results, publication, and selected-Method loading.
- `evaluation` owns canonical self-contained observations and transient reduction.

### Durable object flow

#### Corpus

`CorpusRequest` names an inclusive chain block range and its UUID. FABLE consumes and validates a
completed canonical `corpus.json`/`blocks.parquet` pair. Corpus production is external.

#### Study and artifact

`TuneRequest` contains one `ExperimentSemantics` and a finite nonempty tuple of complete Methods from one model family. `run_candidate()` prepares training state, fits one supplied Method, and appends one successful `RetainedResult` to Study scratch. `publish_study()` renames the ordered result set to its canonical JSON file.

A baseline `TrainRequest` embeds its complete `TrainingDefinition`. A selected-Study request instead names the exact Study UUID and result index while carrying the experiment. Training loads that exact row's Method, composes the definition from the source experiment and Method, fits through Lightning, and renames the native weights-only best checkpoint to the artifact UUID address. The checkpoint embeds the request, feature and target state, and—only for selected-Study training—the exact result index and Method.

#### Evaluation

`EvaluateRequest` names an artifact, same-source Corpus, testing origin window, and evaluation UUID. Evaluation rebuilds historical examples with persisted state, runs the artifact on CUDA, and publishes `evaluation.json` with `observations.parquet`. Each ordered observation stores `h_i`, `hat{k}_i`, `k_i*`, `hat{ell}_i`, `B_i(0)`, `P_i(0)`, `B_i(hat{k}_i)`, `P_i(hat{k}_i)`, and `m_i` under the exact field names in the [reference](#canonical-observations).

`reduce_evaluation()` validates the strict request and UUID, exact schema, nonnull row count, and ordered testing window, then trusts the published row values and returns a transient one-row seven-metric DataFrame. The rolling reduction reads explicit `K=2…5` observation files and returns one transient row of six economic metrics per architecture-chain cell. Neither path loads an artifact or Corpus or persists its result.

### Training and inference

Historical preparation produces lazy datasets over contiguous feature, fee, and block-number backing.

The model union is closed: LSTM, Transformer, or Transformer-LSTM. Every model consumes float32 `[B,C,F]` and returns action logits `[B,K]` plus a scalar standardized minimum-fee prediction `[B]`. The architecture is independent of target construction and evaluation accounting.

Live serving loads cwd-local `SERVING.yaml` once, resolves the artifact storage root from the
`STORAGE_ROOT` environment variable, selects an exact artifact cell, freezes the latest closed
head, reads its `C-1` predecessors, applies the checkpoint's ordered feature state, runs one CPU
batch, and returns the decoded target coordinate and positive finite raw minimum-fee prediction.

### External boundaries

Corpus production is external to FABLE. Live serving uses ordinary Web3 RPC clients supplied by
cwd-local configuration.

`fable.execution.submit()` is the boundary to native OpenSSH and Slurm execution. [ADR 0007](docs/adr/0007-native-external-execution-boundary.md) records that ownership decision.

Completed objects have direct canonical addresses and own their exact requests once. UUIDs provide instance identity; associations provide meaning. See [ADR 0006](docs/adr/0006-direct-durable-object-authority.md).

## One decision, end to end

FABLE makes a decision immediately after a closed parent block `h`. Every number in this hand-computable Ethereum example is a fabricated teaching value.

### 1. Fix the geometry

Suppose:

```text
h = 25,400,000
C = 200 closed context blocks
K = 5 future outcome blocks
```

The model may see exactly blocks `h-C+1 … h`, or `25,399,801 … 25,400,000`. The complete outcome is `h+1 … h+K`. Actions are zero-based:

| Action `k` | Intended target block |
| ---: | ---: |
| 0 | 25,400,001 |
| 1 | 25,400,002 |
| 2 | 25,400,003 |
| 3 | 25,400,004 |
| 4 | 25,400,005 |

The arithmetic is always `target_block = h + 1 + k`.

### 2. Build only closed-parent inputs

For this calculation only, let the request's ordered feature tuple contain three supported features. Suppose closed parent `h` has:

```text
base_fee_per_gas = 24,000,000,000 wei/gas
gas_used         = 27,000,000 gas
gas_limit        = 36,000,000 gas
```

The raw closed-row features are:

```text
log_base_fee_per_gas = ln(24,000,000,000 / (1 wei/gas))
                     = 23.901320

gas_utilization = 27,000,000 / 36,000,000
                = 0.75
```

Ethereum's forming-child fee follows the exact parent recurrence. The parent target is `36,000,000 // 2 = 18,000,000 gas`. Usage exceeds target by `9,000,000`, so ordered integer arithmetic gives:

```text
increase = 24,000,000,000 * 9,000,000 // 18,000,000 // 8
         = 1,500,000,000 wei/gas

forming_child_base_fee = 25,500,000,000 wei/gas

log_exact_forming_base_fee_per_gas
  = ln(25,500,000,000 / (1 wei/gas))
  = 23.961944
```

All feature inputs come from block `h` or earlier. The exact child fee is an Ethereum parent-state result. The other 199 rows are prepared the same way from their own closed facts.

Training-only Float64 means and population standard deviations standardize the ordered raw matrix. The one-origin input is finite float32 `[C,F] = [200,3]`; a live batch is `[1,200,3]`.

### 3. Keep outcomes on the other side of the origin

Invent these complete future base fees:

```text
h+1 ... h+5 = [25.5, 23, 21, 20, 22] gwei/gas
```

They are stored and compared as positive Int64 wei/gas:

```text
[25_500_000_000, 23_000_000_000, 21_000_000_000, 20_000_000_000, 22_000_000_000]
```

Using the [canonical notation](#decision-and-target-notation), the raw outcomes are
`B_i(0) … B_i(4)`, the label is `k_i*=3`, and `m_i=20,000,000,000 wei/gas`.

The dataset item is:

| Value | Shape and dtype |
| --- | --- |
| `inputs` | `[200,3]`, float32 |
| `label` | scalar, int64 |
| `target` | scalar, float32 |
| `base_fees` | `[5]`, int64 |
| `origin_block` | scalar, int64 |

The raw minimum first enters Float64 natural-log coordinates:

```text
ell_i = ln(20,000,000,000 / (1 wei/gas)) = 23.718998
```

For a purely illustrative fitted `TargetState(mean=23.5, standard_deviation=0.25)`:

```text
mu_ell    = 23.5
sigma_ell = 0.25
z_i       = (23.718998 - 23.5) / 0.25 = 0.875992
```

Real state is fitted once from all retained training-origin minima with Float64 `ddof=0`. Validation, testing, and live inference use the persisted state.

### 4. Separate the roles

Every retained origin must have its complete `K`-block outcome inside its role. If validation begins at parent block `V`, a training origin is eligible only when `h+K < V`. Testing starts only after `validation_last_parent + K`.

Training fits feature state, target state, and weights. Validation selects epochs and retained candidate objectives. Testing measures held-out behavior.

### 5. Compute one two-head loss

For one origin, suppose the model returns:

```text
action_logits = [0.2, 1.1, -0.1, 1.7, 0.5]
minimum_fee_z = 0.7
```

The five `action_logits` values are `a_i0 … a_i4`. With `k_i*=3`, cross-entropy is:

```text
CE = log(sum_{k in 𝒦}(exp(a_ik))) - a_i,k_i*
   = log(exp(0.2)+exp(1.1)+exp(-0.1)+exp(1.7)+exp(0.5)) - 1.7
   ≈ 0.805777
```

Here `hat{z}_i=0.7`, so `e_i=hat{z}_i-z_i=0.7-0.875992=-0.175992`. Native
Smooth L1 uses its default transition at one standardized-target unit, so `|e_i| < 1`:

```text
SmoothL1(e_i) = 0.5 * e_i^2 ≈ 0.015487
total         = CE + SmoothL1(e_i) ≈ 0.821264
```

For this one-origin batch, `mean_total = total`. In a larger batch every origin contributes native unweighted cross-entropy plus native default Smooth L1 once, with sample count `B` as the denominator. No loss definition, mode, scale, threshold, or fitted classification state is request or artifact authority.

### 6. Decode and evaluate

Canonical decode gives `hat{k}_i=3`. The intended target is block `25,400,004`.

For this outcome:

```text
B_i(0)        = 25.5 gwei/gas
B_i(hat{k}_i) = B_i(3) = 20.0 gwei/gas
P_i(0)        = 2.0 gwei/gas
P_i(hat{k}_i) = P_i(3) = 4.0 gwei/gas
k_i*          = 3
m_i           = B_i(k_i*) = 20.0 gwei/gas
```

The durable observation stores:

```text
origin_block                         = 25,400,000
predicted_action_k                   = 3
predicted_minimum_log_base_fee       = 23.675
minimum_action_k                     = 3
immediate_base_fee_per_gas           = 25,500,000,000
immediate_effective_priority_fee_per_gas_p50 = 2,000,000,000
selected_base_fee_per_gas            = 20,000,000,000
selected_effective_priority_fee_per_gas_p50  = 4,000,000,000
minimum_base_fee_per_gas             = 20,000,000,000
```

Reduction uses this row directly. Base-fee savings is `(25.5-20.0)/25.5 ≈ 0.215686`; P50 fee-inclusive savings is `1-((20.0+4.0)/(25.5+2.0)) ≈ 0.127273`; and the optimality gap is `(20.0-20.0)/20.0 = 0`. The absolute natural-log error is about `0.043998` and the squared error about `0.001936`. No losses, timestamps, waits, horizons, standardized predictions, or derived metrics are stored in the observation.

### 7. Carry the same contract into serving

The checkpoint fixes chain association, `C`, `K`, ordered features, feature state, target state, model definition, and weights. Live serving freezes the latest closed head, fetches exactly `C-1` predecessors, creates `[1,C,F]`, validates the action logits, and decodes the same way.

Continuing the teaching values, the API response shape is:

```json
{
  "head_block": 25400000,
  "selected_action_k": 3,
  "target_block": 25400004,
  "predicted_minimum_base_fee_per_gas": 19139115255.738445
}
```

The last value follows current serving arithmetic: the displayed float32 `0.7` is
`0.699999988079071` as a Python float, then
`u * exp(hat{ell}_i) = (1 wei/gas) * exp(23.5 + 0.25 * 0.699999988079071)`.

## Scientific contract

FABLE is a closed-parent, fixed-block-horizon temporal learning system. This document owns the causal information set, `C/K/k` geometry, fitted-state rules, feature and target equations, evaluation estimands, claim boundaries, sources, and limitations.

### Lineage and ownership

The manuscript *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments* describes a broader spatial, temporal, and distributed-reputation system. Its temporal experiment motivates a future minimum-block decision, an associated scalar fee prediction, the LSTM/Transformer/Transformer-LSTM comparison, chronological roles, and a weighted cross-entropy plus Smooth-L1 lineage.

FABLE specifies the current closed-parent origins, fixed block-count geometry, causal features,
raw-integer target selection, training-fitted state, fixed training loss, exhaustive equal-origin
evaluation, durable objects, and serving semantics.

### Closed-parent causality

A decision origin occurs immediately after block `h` closes. Facts in blocks through `h` may be inputs. Facts from `h+1` onward are outcomes and cannot influence features or fitted state available at that origin.

For context length `C` and horizon `K`:

```text
context rows:  h-C+1, ..., h
outcome rows:  h+1,   ..., h+K
actions:       k in {0, ..., K-1}
target block:  b = h+1+k
```

Block number owns geometry. Timestamp spacing may vary while the number of context and outcome rows stays fixed.

`C` and `K` are generic positive request values. Python owns no named study matrix, ordering, or staged stopping policy; external orchestration supplies actual runs, and persisted requests and artifacts record what ran.

An origin is eligible only with all `C` context rows and all `K` outcome rows. At a boundary where the next role begins at parent `V`, an earlier origin must satisfy `h+K < V`. Therefore no training outcome reaches validation, and no validation outcome reaches testing.

### Decision and target notation

For origin `i` with closed parent `h_i`, the canonical decision and scalar-target notation is:

```text
𝒦          = {0, ..., K-1}
B_i(k)     = raw base fee at h_i+1+k, for k in 𝒦
k_i*       = argmin_{k in 𝒦} B_i(k), defined as the smallest minimizing k
m_i        = B_i(k_i*) = min_{k in 𝒦} B_i(k)
u          = 1 wei/gas
ell_i      = ln(m_i / u)
mu_ell     = mean_Float64(ell_i over retained training origins)
sigma_ell  = std_Float64(ell_i over retained training origins, ddof=0)
z_i        = Float32((ell_i - mu_ell) / sigma_ell)
a_ik       = action logit for k in 𝒦
hat{k}_i   = argmax_{k in 𝒦} a_ik
hat{z}_i   = predicted standardized log minimum
hat{ell}_i = mu_ell + sigma_ell * hat{z}_i
```

Raw Int64 fee comparison defines `k_i*` and `m_i` before any floating conversion. The fitted
`sigma_ell` must be positive. Because `m_i/u` is a ratio of like units, `ell_i` and `hat{ell}_i`
are dimensionless log coordinates.

### Role ownership and fitted populations

Training alone may fit:

- feature population means and standard deviations;
- target natural-log mean and standard deviation;
- neural weights.

Validation selects the earliest best epoch and supplies candidate objectives. Testing measures only. Changing a method, feature route, horizon, context, or other scientific decision after inspecting testing would turn that measurement into selection evidence.

#### Feature state

Let raw training-support feature row `x_r ∈ R^F`. For each ordered feature `j`:

```text
mu_j    = (1/N) sum_r x_rj
sigma_j = sqrt((1/N) sum_r (x_rj - mu_j)^2)
z_rj    = (x_rj - mu_j) / sigma_j
```

Fitting uses Float64 and `ddof=0`; every `sigma_j` must be positive. Transformation returns finite float32. Training support contains each closed block row once, so overlapping model windows do not reweight feature-state fitting.

### Causal features

The request supplies a nonempty unique ordered tuple drawn from the supported names, making feature choice request-authored.

| Feature | Raw equation and unit | Domain and availability |
| --- | --- | --- |
| `log_base_fee_per_gas` | `ln(base_fee_per_gas / (1 wei/gas))` | Fee positive; closed-row header fact. |
| `gas_utilization` | `gas_used / gas_limit` | `gas_limit>0`, `0≤gas_used≤gas_limit`; known after row close. |
| `log_exact_forming_base_fee_per_gas` | `ln(exact_child_base_fee / (1 wei/gas))` | Positive; Ethereum-only parent-state recurrence. |
| `log_gas_limit` | `ln(gas_limit / (1 gas))` | Gas limit positive; closed-row header fact. |
| `log1p_tx_count` | `ln(1 + tx_count / (1 transaction))` | Transaction count nonnegative; known after row close. |
| `log1p_effective_priority_fee_per_gas_p50` | `ln(1 + effective_priority_fee_per_gas_p50 / (1 wei/gas))` | P50 nonnegative; included-transaction closed-row fact. |
| `block_interval_seconds` | `timestamp_b - timestamp_{b-1}` seconds | Positive; requires the real predecessor row. |
| `hour_sin` | `sin(2π hour_UTC/24)` | `hour_UTC = (timestamp//3600) mod 24`; closed timestamp. |
| `hour_cos` | `cos(2π hour_UTC/24)` | Same angle and availability. |

The exact forming-fee column implements the Ethereum parent-known recurrence. Polygon and Avalanche requests use the other supported features.

#### Ethereum forming-child recurrence

For positive parent fee `f`, parent gas used `g`, and positive gas limit `L`, use Python integers throughout:

```text
t = L // 2

if g == t:
    f_child = f
elif g > t:
    f_child = f + max(f * (g - t) // t // 8, 1)
else:
    f_child = f - f * (t - g) // t // 8
```

`t` and the final child fee must be positive. Python integers carry the recurrence through the two ordered divisions; the one-wei floor applies only upward. The completed positive integer is then logged in Float64. This follows the integer ordering in [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559#specification).

### Historical tensors and targets

One lazy historical item has:

| Value | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `inputs` | `[C,F]` | float32 | Standardized closed rows `h_i-C+1 … h_i`. |
| `label` | scalar | int64 | Canonical `k_i*`. |
| `target` | scalar | float32 | Canonical `z_i`. |
| `base_fees` | `[K]` | int64 | `[B_i(0), ..., B_i(K-1)]`. |
| `origin_block` | scalar | int64 | Closed parent `h_i`. |

Collation produces `[B,C,F]`, `[B]`, `[B]`, `[B,K]`, and `[B]`.

For origin `i`, the positive Int64 outcome vector is
`[B_i(0), ..., B_i(K-1)]`. The [canonical notation](#decision-and-target-notation) defines its
`k_i*`, `m_i`, `ell_i`, and `z_i`; NumPy [`argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
implements that raw-integer action selection.

Target state is fitted over retained training origins only. Float64 mean and `ddof=0` population
standard deviation produce canonical `mu_ell` and positive `sigma_ell`; transformation follows
the canonical `z_i` equation exactly and returns float32.

### Targets, loss, and decode

All concrete model definitions return:

```text
action_logits: [B,K]
minimum_fee_z: [B]
```

The first head supplies canonical logits `a_ik`. The second predicts canonical `hat{z}_i`, the
standardized dimensionless log of the same horizon minimum.

#### Classification

For origin `i`, using its logits `a_i0 … a_i,K-1` and label `k_i*`:

```text
c_i = CE([a_i0, ..., a_i,K-1], k_i*)
```

Classification is native unweighted cross-entropy. It has no weighting mode, scale, fitted support, or configuration field.

#### Regression

Regression is native Smooth L1 with its default transition at one standardized-target unit. Its
error is `e_i=hat{z}_i-z_i`:

```text
smooth_l1(e_i) = 0.5 e_i^2       if |e_i| < 1
                 |e_i| - 0.5     otherwise

r_i = smooth_l1(e_i)
```

#### Total

```text
t_i = c_i + r_i
mean_total = (sum_i t_i) / B
```

The denominator is the number of origins in the batch. These are training and validation losses only. The operative functions match PyTorch's [`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) and [`smooth_l1_loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html).

Decode implements canonical `hat{k}_i` with native `argmax(action_logits, dim=-1)`. Equal maximum
logits select the first index, and decode depends on the logits alone.

### Model concepts

FABLE uses a closed discriminated union of three concrete sequence models:

- LSTM recurrently summarizes the fixed context and uses its final state.
- Transformer projects each row, adds sinusoidal positions, applies self-attention, and uses the final encoded position.
- Transformer-LSTM applies the Transformer encoder, then recurrently summarizes the encoded sequence.

All three attach the same two MLP heads. Architecture capacity belongs to `ModelDefinition` or Method; target and loss meaning stays in `fable.min_block_fee`.

### Evaluation estimands

For testing origin `i`, the canonical observation stores these direct values:

```text
hat{k}_i        predicted action
k_i*            minimum action
B_i(0)          immediate base fee
B_i(hat{k}_i)   selected base fee
P_i(0)          included-transaction effective-priority-fee P50 in block h_i+1
P_i(hat{k}_i)   included-transaction effective-priority-fee P50 in block h_i+1+hat{k}_i
m_i             minimum base fee
hat{ell}_i      predicted dimensionless log minimum
```

Evaluation de-standardizes `hat{z}_i` to `hat{ell}_i` before publication. Reduction reads the
stored facts directly; `ell_i=ln(m_i/u)` is the true dimensionless log coordinate.

Over the testing origins, reduction returns exactly seven Float64 metrics:

```text
accuracy                = mean_i indicator[hat{k}_i = k_i*]
log_fee_mae             = mean_i |hat{ell}_i - ell_i|
log_fee_mse             = mean_i (hat{ell}_i - ell_i)^2
base_fee_savings        = mean_i ((B_i(0) - B_i(hat{k}_i)) / B_i(0))
p50_fee_inclusive_savings
                        = mean_i (1 - (B_i(hat{k}_i) + P_i(hat{k}_i))
                                          / (B_i(0) + P_i(0)))
base_fee_optimality_gap = mean_i ((B_i(hat{k}_i) - m_i) / m_i)
```

`f1_macro` is standard unweighted macro-F1 over the union of action classes present in truth or predictions, with zero division equal to zero. Classes absent from both do not enter the mean.

All three economic metrics are mean per-origin fractions, not ratios of fee sums. Positive base fees make their denominators defined. Both savings metrics are higher-is-better; `base_fee_savings` remains base-fee-only, while `p50_fee_inclusive_savings` is a retrospective representative-cost proxy using each outcome block's included-transaction P50, not an inclusion guarantee. `base_fee_optimality_gap` is nonnegative and lower is better. Natural-log errors compare dimensionless coordinates relative to `u=1 wei/gas` and lower is better. Accuracy and macro-F1 are unitless and higher is better. Economic values remain fractions for later percentage formatting.

#### Fixed-deadline rolling comparison

The rolling comparison fixes each `K=5` origin `h` and deadline `D=h+5`. It aligns the completed predictions at `h` for `K=5`, `h+1` for `K=4`, `h+2` for `K=3`, and `h+3` for `K=2`. At each stage, `k=0` selects the next block and stops; otherwise the next shorter-horizon prediction is consulted. If the `K=2` prediction also waits, it must be `k=1` and selects `D`. The deadline never moves.

Precomputed predictions preserve historical causality because a later row affects the result only when every earlier stage waited. The comparison performs no model inference or Corpus hydration.

For every architecture-chain cell, the rolling reduction uses every `K=5` testing origin at stride one and returns one-shot and rolling values for base-fee savings, P50 fee-inclusive savings, and base-fee optimality gap. Savings use the immediate action as their baseline; optimality gap uses the earliest minimum within the original five-block window. These are exact descriptive means for the sealed held-out period. There is no bootstrap, confidence interval, significance test, resampling, stride thinning, or persisted pairwise delta.

### HPO interpretation

A `TuneRequest` freezes the experiment and one finite tuple of complete Methods. An operator submits complete Methods from that tuple. Each successful fit contributes validation total loss, earliest best epoch, and completed epochs in retention order. Selected training names an exact result index.

The planned thesis protocol is staged: leave one feature group out on validation, run the `C` study, run HPO on the winning feature/context route, then run `K` sensitivity and final testing. Fit methods use seed `2026`. This is not a full factorial design. No experiments have been run.

## Architecture and deep interfaces

The sections below place each direct owner interface beside the scientific and durable-object contracts it serves. Exact public records, paths, commands, YAML fields, and schemas remain in [Exact reference](#exact-reference).

### Corpus input

FABLE consumes and validates one completed canonical Corpus pair. Corpus production is external.

```text
corpora/<corpus_id>/
  corpus.json
  blocks.parquet
```

`corpus.json` stores the exact `CorpusRequest` and one finalized anchor. `blocks.parquet` stores the
requested contiguous rows in block-number order with the exact eight-column canonical schema
documented in the [reference](#corpus-object). `load_corpus()` strictly hydrates the request and
anchor, checks the requested UUID, constructs the canonical `BlockFrame`, and requires the anchor
to cover the completed range.

`BlockFrame(frame, definition)` is the public canonical-row boundary. Its `definition` identifies the exact owned range, `select_range(first_block, last_block)` returns an inclusive trusted subrange, and `to_polars()` returns an isolated native frame. Construction and native access isolate caller mutation. Range selection is positional after the full frame has been validated; it does not rescan rows. The value carries neither hashes nor finality provenance.

### Temporal preparation

Temporal preparation has two direct paths: historical fixed-block examples and live closed-head inference. Both use the same ordered feature functions and persisted training-only feature state.

#### Historical interface

`prepare_fit_history(corpus, experiment)` validates complete context and outcome support, fits state from training support only, and returns:

```text
HistoricalPreparation
  training: HistoricalDataset
  validation: HistoricalDataset
  feature_state: FeatureState
  target_state: TargetState
```

`prepare_historical_window(corpus, experiment, window, *, feature_state, target_state)` prepares an exact testing window with persisted state. Testing must begin after all validation outcomes are complete.

For origin `h`, support is exact by block number:

```text
context:  h-C+1 ... h
outcome:  h+1   ... h+K
action k: target h+1+k
```

The Corpus must include every context and outcome block named by this geometry.

#### Lazy dataset

Preparation builds one contiguous CPU backing over the needed range:

- transformed feature rows: float32 `[rows,F]`;
- raw base fees: int64 `[rows]`;
- block numbers: int64 `[rows]`.

It stores per-origin row positions, canonical `k_i*` labels as int64 vectors, and standardized
`z_i` targets as a float32 vector. `HistoricalDataset.__getitem__()` slices one float32 `[C,F]`
input and one int64 `[K]` raw fee outcome on demand, plus scalar int64 label and origin block and
scalar float32 target.

#### Feature state

The ordered feature tuple is request authority. Names must be unique and supported by the direct implementation. Raw features are assembled in exactly that order as Float64. Training-support population means and standard deviations use `ddof=0`; a constant feature is invalid. Transform applies those values and returns finite C-contiguous float32 rows.

Exact formulas, units, causal availability, and the Ethereum forming-fee recurrence belong to the [theory](#causal-features).

#### Outcome preparation

Historical outcomes remain positive int64 `B_i(k)` fees. For each origin, canonical `k_i*` is the
label, and `m_i` feeds the fitted target state.

Role boundaries are complete-outcome boundaries. The training last parent plus `K` must be strictly before the first validation parent; an authored testing window obeys the same rule after validation. Training alone fits feature state, target state, and model weights.

#### Live interface

Serving freezes one latest closed head `h`, materializes the same exact eight-column `BlockFrame` used by historical preparation, transforms the ordered features with the artifact's `FeatureState`, and constructs float32 `[1,C,F]`. Historical preparation owns outcomes, labels, and target values. The exact live RPC and app boundary is in [Serving and mobile](#serving-and-mobile).

The artifact fixes `C`, `K`, feature order, and fitted states. Decoding returns `hat{k}_i`, so
serving reports `h_i+1+hat{k}_i` as the target block coordinate. It de-standardizes `hat{z}_i` to
`hat{ell}_i` with the artifact's `TargetState` and returns `u * exp(hat{ell}_i)` as the positive
finite `predicted_minimum_base_fee_per_gas`.

### Minimum-block-fee task

Top-level `fable.min_block_fee` keeps the architecture-neutral target, loss, and decode contract. Temporal preparation supplies its targets, model families return its output, and evaluation consumes the result.

#### Owned values

`TargetState` contains canonical `mu_ell` and positive `sigma_ell`, the Float64 population state of
dimensionless `ell_i=ln(m_i/u)` over retained training origins.

`MinBlockFeeOutput` has two tensors:

```text
action_logits:  [B,K]
minimum_fee_z:  [B]
```

The scalar head predicts canonical `hat{z}_i`, the standardized dimensionless log of the horizon
minimum. Its scientific interpretation is defined in the [theory](#targets-loss-and-decode).

#### Direct functions

- `fit_target_state(raw_minima)` requires a nonempty positive int64 vector, computes Float64
  `ell_i`, `mu_ell`, and `sigma_ell` with `ddof=0`, and rejects constant targets.
- `standardize_target(raw_minima, state)` returns finite contiguous float32 `z_i` values.
- `min_block_fee_loss(...)` trusts tensors owned by model and historical-preparation internals. It directly computes native unweighted cross-entropy and native default Smooth L1 per origin, returning detached per-origin totals and the gradient-carrying sample-denominator mean.
- `decode_action(output)` applies the canonical action decode.

The exact equations are in the [theory](#targets-loss-and-decode).

#### Boundaries

Temporal preparation owns raw `[K]` `B_i(k)` outcomes, `k_i*` labels, and standardized `z_i`
targets. Model code owns the sequence encoder and the two concrete heads. Evaluation owns
observation publication and economic accounting.

### Study

Tuning is a bounded question over a finite tuple of complete Methods. A Study contains the exact `TuneRequest` and its ordered successful results.

#### Request and membership

`TuneRequest` fixes a Study UUID, Corpus UUID, `ExperimentSemantics`, and a nonempty tuple of unique complete Methods. Every Method uses the same model family and owns one `ModelDefinition` plus its complete fit policy.

`apply_method(request, method)` requires exact whole-Method membership in `request.methods`, then returns `TrainingDefinition(experiment=request.experiment, method=method)`.

#### Candidate run

`run_candidate(storage_root, request, method, deployment)` loads the request's Corpus, prepares training history and state, fits the exact Method through native Lightning, and retains one successful result. Candidate checkpoints stay in Study scratch; training publishes artifacts.

`RetainedResult` has four fields:

- the exact Method;
- finite complete-validation total-loss objective;
- one-based earliest selected epoch;
- one-based completed epoch count.

The selected epoch cannot exceed completed epochs, and completed epochs cannot exceed the Method maximum.

#### Ordered progress and publication

Candidate success appends to `studies/.<study_id>/progress.json`. Existing progress must contain the identical request. Appends preserve caller completion order and directly replace the progress file through one hidden temporary sibling.

`publish_study(storage_root, study_id)` validates progress and renames it to `studies/<study_id>.json`, preserving completion order. An existing canonical Study is an error.

#### Selected training

A selected-Study `TrainRequest` supplies the exact Study UUID and zero-based `study_result_index`. `load_selected_method()` strictly loads the canonical Study, verifies Study and Corpus associations, and returns the Method from that ordered row. The artifact association composes its `TrainingDefinition` from the source experiment and returned Method.

The resulting native artifact embeds the same result index and Method for later loading and evaluation.

### Evaluation

Evaluation separates canonical self-contained observations from transient metrics. Explicit UUIDs connect the request, artifact, Corpus, and observations.

#### Canonical evaluation

`evaluate(request, storage_root, deployment)` loads the exact Corpus and native artifact, requires the artifact's source Corpus to equal the evaluation Corpus, prepares the testing origin window with persisted state, and performs CUDA inference.

For every eligible origin the evaluation publisher owns construction of one ordered, nonnull observation containing `h_i`, `hat{k}_i`,
`k_i*`, `hat{ell}_i`, `B_i(0)`, `P_i(0)`, `B_i(hat{k}_i)`, `P_i(hat{k}_i)`, and `m_i` under the canonical field names. Work is
written under `evaluations/.<evaluation_id>/` and renamed to:

```text
evaluations/<evaluation_id>/
  evaluation.json
  observations.parquet
```

The JSON is exactly the `EvaluateRequest`. The parquet schema is the canonical nine-column contract in the [reference](#canonical-observations).

#### Transient reduction

`reduce_evaluation(storage_root, evaluation_id) -> polars.DataFrame` strictly hydrates the request, validates its evaluation UUID, exact Parquet schema, expected nonnull row count, and ordered testing origins, then trusts the publisher-owned row values and reduces only `observations.parquet`. It requires all seven computed metrics to be finite. It does not reload the artifact or Corpus or externally authenticate the horizon or source. The result has no evaluation ID, count, sums, supports, arrays, or auxiliary fields and is not persisted.

The rolling reduction reads only each named Evaluation's `observations.parquet`. Its in-memory roster maps exactly nine human-readable architecture-chain cell names to mappings from horizons `2`, `3`, `4`, and `5` to their Evaluation UUIDs. The final experiment runner owns that scientific association. Reduction verifies exact schemas, nonnull consecutive origins, action ranges, and required shifted coverage. Its nine-row, six-metric DataFrame is transient and is not persisted.

## Exact reference

This reference defines FABLE's strict requests, completed objects, direct addresses, commands, operator YAML, serving/mobile surfaces, and evaluation schemas.

### Scalar conventions

- Object IDs are UUIDv4.
- `PositiveInt` means strict integer `>0`; `NonNegativeInt` means strict integer `≥0`. Booleans are not integers.
- Scientific floats are finite. Positive/nonnegative bounds are stated per field.
- Block ranges and origin windows are inclusive.
- Base fees are positive Int64 wei/gas unless a field explicitly says Float64 aggregation.
- Timestamps and elapsed values are integer seconds.
- Strict records reject unknown fields and revalidate nested instances.

Distribution name, import root, and installed executable are `fable`; the static distribution version is `0.1.0`.

### Requests and definitions

#### Corpus

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `CorpusDefinition` | `chain_id` | PositiveInt |
|  | `first_block` | NonNegativeInt |
|  | `last_block` | NonNegativeInt, `last_block≥first_block` |
| `CorpusRequest` | `corpus_id` | UUIDv4 |
|  | `definition` | `CorpusDefinition` |

#### Scientific semantics

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `BlockWindow` | `first_parent_block` | NonNegativeInt |
|  | `last_parent_block` | NonNegativeInt, not before first |
| `ExperimentSemantics` | `training_window` | `BlockWindow` |
|  | `validation_window` | `BlockWindow` |
|  | `context_blocks` | PositiveInt `C` |
|  | `horizon_blocks` | PositiveInt `K` |
|  | `ordered_features` | nonempty unique tuple of nonempty strings |

The training last parent plus `K` must be strictly less than the validation first parent.

#### Model definitions

`ModelDefinition` is a discriminated union on `family`:

| Family | Ordered fields after `family` |
| --- | --- |
| `lstm` | `hidden: PositiveInt`; `layers: PositiveInt`; `head_hidden: PositiveInt`; `dropout: 0≤float<1` |
| `transformer` | `model_width`; `attention_heads`; `transformer_layers`; `feedforward_width`; `head_hidden`: PositiveInt; `dropout: 0≤float<1` |
| `transformer_lstm` | `model_width`; `attention_heads`; `transformer_layers`; `feedforward_width`; `lstm_hidden`; `lstm_layers`; `head_hidden`: PositiveInt; `dropout: 0≤float<1` |

Transformer widths must be even and divisible by `attention_heads`.

#### Method

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `FitMethod` | `learning_rate` | finite float `>0` |
|  | `weight_decay` | finite float `≥0` |
|  | `accumulation` | PositiveInt |
|  | `gradient_clip_norm` | finite float `≥0` |
|  | `seed` | NonNegativeInt |
|  | `max_epochs` | PositiveInt |
|  | `validate_every_completed_epoch` | PositiveInt |
|  | `patience` | NonNegativeInt |
|  | `min_delta` | finite float `≥0` |

Every serialized `Method` has ordered fields `model: ModelDefinition` and `fit: FitMethod`. A `TuneRequest` owns a nonempty tuple of unique complete Methods and requires every `method.model.family` to match.

#### Study, training, and workflow requests

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `TrainingDefinition` | `experiment` | `ExperimentSemantics` |
|  | `method` | complete `Method` |
| `BaselineSource` | `kind` | exactly `"baseline"` |
|  | `corpus_id` | UUIDv4 |
|  | `training_definition` | `TrainingDefinition` |
| `SelectedStudySource` | `kind` | exactly `"selected_study"` |
|  | `corpus_id` | UUIDv4 |
|  | `study_id` | UUIDv4 |
|  | `study_result_index` | NonNegativeInt |
|  | `experiment` | `ExperimentSemantics` |
| `TrainRequest` | `workflow` | exactly `"train"` |
|  | `artifact_id` | UUIDv4 |
|  | `source` | `BaselineSource | SelectedStudySource` |
| `TuneRequest` | `workflow` | exactly `"tune"` |
|  | `study_id` | UUIDv4 |
|  | `corpus_id` | UUIDv4 |
|  | `experiment` | `ExperimentSemantics` |
|  | `methods` | nonempty unique tuple of complete, same-family Methods |
| `EvaluateRequest` | `workflow` | exactly `"evaluate"` |
|  | `evaluation_id` | UUIDv4 |
|  | `artifact_id` | UUIDv4 |
|  | `corpus_id` | UUIDv4 |
|  | `testing_window` | `BlockWindow`; must follow complete validation outcomes |

`WorkflowRequest` is exactly `TrainRequest | EvaluateRequest`. `TuneRequest` is intentionally separate.

Fresh constructors:

```python
fresh_train_request(source: TrainingSource) -> TrainRequest
fresh_tune_request(
    corpus_id: UUID,
    experiment: ExperimentSemantics,
    methods: tuple[Method, ...],
) -> TuneRequest
fresh_evaluate_request(
    artifact_id: UUID,
    corpus_id: UUID,
    testing_window: BlockWindow,
) -> EvaluateRequest
```

### Durable addresses and objects

Given an explicit `storage_root`:

```text
corpora/<corpus_id>/corpus.json
corpora/<corpus_id>/blocks.parquet
experiments/{feature_ablation,c_study,hpo,k_study}/<UUID>.json
studies/<study_id>.json
artifacts/<artifact_id>.ckpt
evaluations/<evaluation_id>/evaluation.json
evaluations/<evaluation_id>/observations.parquet
```

IDs are lowercase UUID strings produced by `str(UUID)` and appear directly in the paths above.

#### Corpus object

`corpus.json` has exactly:

```text
request: CorpusRequest
finalized_anchor:
  block_number: integer >= 0
  block_hash: 64 lowercase hexadecimal characters
```

`blocks.parquet` has this exact ordered, nonnull schema:

| # | Column | Type | Unit/rule |
| ---: | --- | --- | --- |
| 1 | `block_number` | Int64 | contiguous inclusive request range |
| 2 | `timestamp` | Int64 | nonnegative seconds; nondecreasing |
| 3 | `chain_id` | Int64 | equals request chain |
| 4 | `base_fee_per_gas` | Int64 | positive wei/gas |
| 5 | `gas_used` | Int64 | gas, `0≤used≤limit` |
| 6 | `gas_limit` | Int64 | positive gas |
| 7 | `tx_count` | Int64 | nonnegative transaction count |
| 8 | `effective_priority_fee_per_gas_p50` | Int64 | nonnegative gas-used-weighted P50 among included transactions, wei/gas |

Direct loader:

```python
load_corpus(storage_root: Path, corpus_id: UUID4) -> Corpus
```

`Corpus.blocks` is a `BlockFrame` whose definition equals the request definition. The finalized anchor covers its last block. These in-memory ownership facts do not change the durable JSON or Parquet formats above.

#### Experiment manifest

Each `experiments/{feature_ablation,c_study,hpo,k_study}/<UUID>.json` contains the matching UUIDv4 `experiment_id` and a nonempty ordered `entries` tuple. Each entry has a nonempty `cell` label and at least one canonical `artifact_id`, `study_id`, or `evaluation_id` UUIDv4. Manifests group canonical references only; they do not duplicate metrics, results, or scientific definitions.

`experiments/feature_ablation.py prepare STORAGE_ROOT` authors the frozen 45-cell request
bundle under `experiments/feature_ablation/.<experiment_id>/`. The ordinary Study CLI submits
its cells. After all canonical Studies exist, `select STORAGE_ROOT EXPERIMENT_ID` reports the
validation-selected feature set per chain, publishes the canonical manifest, and removes the
temporary bundle. The script does not submit jobs or persist copied metrics.

#### Study object

`studies/<study_id>.json` is a strict `Study`:

```text
request: TuneRequest
trials: nonempty ordered tuple[RetainedResult, ...]
```

Each `RetainedResult` has exact ordered fields:

| Field | Type/rule |
| --- | --- |
| `method` | exact complete Method contained in `request.methods` |
| `objective` | finite float validation total loss |
| `selected_epoch` | integer `≥1` |
| `completed_epochs` | integer `≥selected_epoch` and `≤method.fit.max_epochs` |

#### Native Lightning artifact

`artifacts/<artifact_id>.ckpt` is the native Lightning weights-only best checkpoint. Its `ArtifactAssociation` contains:

| Ordered field | Type/rule |
| --- | --- |
| `request` | exact `TrainRequest`; embedded artifact UUID must match path |
| `feature_state` | Float64 means and positive standard deviations, equal feature width |
| `target_state` | Float64 finite mean and positive standard deviation |
| `study_result_index` | absent/null for baseline; exact nonnegative source index for selected Study |
| `method` | absent/null for baseline; exact selected Method for selected Study |

Direct loader:

```python
load_artifact(
    storage_root: Path,
    artifact_id: UUID,
) -> tuple[ArtifactAssociation, torch.nn.Module]
```

#### Evaluation object

`evaluation.json` is exactly the `EvaluateRequest`. `observations.parquet` is the canonical schema below. Reductions are transient views over this directory.

### CLI

Three public command leaves:

```text
fable submit REQUEST.json [REQUEST.json ...]
fable study run TUNE_REQUEST.json METHOD.json
fable study finalize STUDY_ID
```

- `submit` accepts one or more WorkflowRequest files and prints one positive Slurm job ID per request.
- `study run` validates one strict TuneRequest and one strict Method, then prints the candidate Slurm job ID.
- `study finalize` requires UUIDv4, reads absolute `STORAGE_ROOT`, and publishes existing progress.

Two help-hidden generated-job leaves:

```text
fable remote workflow
fable remote candidate
```

Generated Slurm scripts call these leaves with strict JSON envelopes on standard input.

### Remote submission

`from fable.execution import submit` is the public execution seam:

```python
submit(request: WorkflowRequest) -> int
```

It reads cwd-local `REMOTE.yaml` with this exact strict schema:

| Section | Ordered field | Type/rule |
| --- | --- | --- |
| root | `ssh` | nonempty string passed to OpenSSH |
|  | `executable` | nonempty absolute installed-executable path |
|  | `storage_root` | nonempty absolute path |
|  | `log_root` | nonempty absolute path |
| `resources` | `partition` | nonempty string |
|  | `gres` | string |
|  | `cpus_per_task` | PositiveInt |
|  | `memory_gb` | PositiveInt, rendered as `--mem=<n>G` |
|  | `time_limit` | nonempty Slurm time string |
| `deployment` | `evaluation_batch_size` | PositiveInt |
|  | `num_workers` | NonNegativeInt |
|  | `pin_memory` | bool |
|  | `prefetch_factor` | PositiveInt or null |
|  | `persistent_workers` | bool |
|  | `deterministic` | bool or exactly `"warn"` |
|  | `benchmark` | bool |
|  | `float32_matmul_precision` | `"highest" | "high"` |
|  | `cuda_matmul_allow_tf32` | bool |
|  | `cudnn_allow_tf32` | bool |

The generated script requests one node/task, writes `%j.out` under `log_root`, exports `STORAGE_ROOT`, and executes `fable remote workflow` or `fable remote candidate` with a stdin heredoc. Submission is one `ssh -T -o BatchMode=yes … sbatch --parsable` call.

`STORAGE_ROOT` is the neutral implicit environment input to current CLI and remote Python paths.
Serving RPC endpoints arrive through cwd-local `SERVING.yaml` values.

### Serving and mobile

The serving factory is `fable.serving:create_app`; its display title is `FABLE Inference API`. It reads cwd-local `SERVING.yaml` once during application lifespan and uses its values literally.

The strict `SERVING.yaml` root contains exactly three records: `ethereum`, `polygon`, and
`avalanche`. Each chain record contains a nonempty `rpc_url` and exactly four UUIDv4 fields:
`k2_artifact_id`, `k3_artifact_id`, `k4_artifact_id`, and `k5_artifact_id`. All fields are required;
extra fields are rejected at both levels. Artifact storage comes from the absolute `STORAGE_ROOT`
environment variable, not from `SERVING.yaml`.

Serving expects Ethereum chain ID `1`, Polygon `137`, and Avalanche C-Chain `43114`; the Polygon and Avalanche clients install the PoA extra-data middleware.

The strict request is:

```text
chain: "ethereum" | "polygon" | "avalanche"
K: 2 | 3 | 4 | 5
```

`POST /inference` returns three nonnegative integers and one positive finite float:

```text
head_block                                = h_i
selected_action_k                         = hat{k}_i
target_block                              = h_i + 1 + hat{k}_i
predicted_minimum_base_fee_per_gas        = u * exp(hat{ell}_i)
```

OpenAPI and interactive documentation routes are disabled. The server verifies provider chain ID, loads the exact selected-Study artifact, requires request `K` to equal artifact `K`, reads the latest head plus `C-1` predecessors, and makes one `eth_feeHistory(C, h, [50])` request. It builds the canonical eight-column `BlockFrame`, prepares float32 `[1,C,F]`, and runs CPU inference. P50 is carried for schema completeness; the current selected features, response, and app do not use it. App use is deferred.

The private Expo app manifest is fixed at:

| Fact | Value |
| --- | --- |
| display | `FABLE Demo` |
| package | `fable-mobile-demo` |
| Expo slug/scheme | `fable-demo` |
| iOS bundle ID | `dev.edoski.fable.demo` |
| Android package | `dev.edoski.fable.demo` |
| entry | `expo/AppEntry` |

Its only backend variable is `EXPO_PUBLIC_FABLE_BACKEND_URL`. It posts the strict request to
`/inference` and displays all four response fields.

### Evaluation API

`Deployment` is the shared strict process-input record exported from `fable.config`:

```python
class Deployment:
    evaluation_batch_size: PositiveInt
    num_workers: NonNegativeInt
    pin_memory: bool
    prefetch_factor: PositiveInt | None
    persistent_workers: bool
    deterministic: bool | Literal["warn"]
    benchmark: bool
    float32_matmul_precision: Literal["highest", "high"]
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
```

Public exports from `fable.evaluation`:

```python
evaluate(
    request: EvaluateRequest,
    storage_root: Path,
    deployment: Deployment,
) -> None

reduce_evaluation(
    storage_root: Path,
    evaluation_id: UUID,
) -> polars.DataFrame
```

#### Canonical observations

Destination: `evaluations/<evaluation_id>/observations.parquet`. Status: canonical, ordered, nonnull, one row per inclusive origin in ascending block order.

| # | Field | Type | Unit/meaning |
| ---: | --- | --- | --- |
| 1 | `origin_block` | Int64 | closed parent `h_i` |
| 2 | `predicted_action_k` | Int64 | decoded `hat{k}_i` |
| 3 | `predicted_minimum_log_base_fee` | Float64 | dimensionless predicted log-minimum coordinate `hat{ell}_i` relative to `u` |
| 4 | `minimum_action_k` | Int64 | canonical `k_i*` |
| 5 | `immediate_base_fee_per_gas` | Int64 | `B_i(0)`, wei/gas |
| 6 | `immediate_effective_priority_fee_per_gas_p50` | Int64 | `P_i(0)`, wei/gas |
| 7 | `selected_base_fee_per_gas` | Int64 | `B_i(hat{k}_i)`, wei/gas |
| 8 | `selected_effective_priority_fee_per_gas_p50` | Int64 | `P_i(hat{k}_i)`, wei/gas |
| 9 | `minimum_base_fee_per_gas` | Int64 | `m_i`, wei/gas |

The file contains predictions and the observed truth needed for local reduction. Losses, timestamps, waits, horizons, standardized predictions, and derived metrics remain absent.

#### Transient reduction

Destination: none. `reduce_evaluation()` structurally validates the completed evaluation, trusts its publisher-owned observation values, and returns a one-row DataFrame whose seven metrics must be finite. Status: derived, transient, noncanonical, nonnull. The row does not store `evaluation_id`, `n`, counts, sums, supports, arrays, or auxiliary fields.

| # | Field | Type | Unit/direction |
| ---: | --- | --- | --- |
| 1 | `accuracy` | Float64 | unitless; higher is better |
| 2 | `f1_macro` | Float64 | unitless; higher is better |
| 3 | `log_fee_mae` | Float64 | dimensionless natural-log error relative to `u`; lower is better |
| 4 | `log_fee_mse` | Float64 | squared dimensionless natural-log error; lower is better |
| 5 | `base_fee_savings` | Float64 | mean per-origin fraction versus immediate; higher is better |
| 6 | `p50_fee_inclusive_savings` | Float64 | mean per-origin representative-cost fraction versus immediate; higher is better |
| 7 | `base_fee_optimality_gap` | Float64 | mean per-origin fraction above optimum; lower is better |

`accuracy` compares `predicted_action_k` (`hat{k}_i`) with `minimum_action_k` (`k_i*`). `f1_macro`
averages over the union of classes appearing in truth or predictions with zero division zero.
Regression compares `predicted_minimum_log_base_fee` (`hat{ell}_i`) with
`ln(minimum_base_fee_per_gas / u)` (`ell_i`). Economic fields are fractions, not percentages or
ratios of sums. `p50_fee_inclusive_savings` is retrospective and does not claim inclusion.

#### Rolling comparison result

Destination: none. The rolling reduction returns one row per explicit architecture-chain cell and requires exactly nine rows. Status: derived, transient, noncanonical, nonnull.

| # | Field | Type | Unit/direction |
| ---: | --- | --- | --- |
| 1 | `cell` | String | operator-authored architecture-chain label |
| 2 | `one_shot_base_fee_savings` | Float64 | mean per-origin fraction versus immediate; higher is better |
| 3 | `rolling_base_fee_savings` | Float64 | mean per-origin fraction versus immediate; higher is better |
| 4 | `one_shot_p50_fee_inclusive_savings` | Float64 | mean per-origin representative-cost fraction versus immediate; higher is better |
| 5 | `rolling_p50_fee_inclusive_savings` | Float64 | mean per-origin representative-cost fraction versus immediate; higher is better |
| 6 | `one_shot_base_fee_optimality_gap` | Float64 | mean per-origin fraction above the five-block optimum; lower is better |
| 7 | `rolling_base_fee_optimality_gap` | Float64 | mean per-origin fraction above the five-block optimum; lower is better |

## Limitations and sources

### Claim boundary and limitations

Evaluation describes target block base fee per gas over every eligible origin in one declared historical window. Its claims are bounded as follows:

- The target and `base_fee_savings` omit priority fee and transaction gas use; `p50_fee_inclusive_savings` adds an included-transaction P50 proxy, not an actual transaction quote.
- Target-block intent does not guarantee inclusion at that block.
- The auxiliary head is not calibrated uncertainty or a quote.
- One seed or one time range does not establish seed, regime, or future robustness.
- Different `K` values are different classification problems; testing cannot choose a best `K`.
- Native assets, fee levels, protocol rules, and ranges differ by chain; totals are never pooled across chains.
- Exhaustive origins remove sampling within the declared range, not temporal dependence or selection bias outside it.

### Sources

- [EIP-1559 specification](https://eips.ethereum.org/EIPS/eip-1559)
- [Reference temporal-model repository at the frozen commit](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/tree/bcf80b92877941e3b05a7dc5138560ffe41df27e)
- [Hochreiter and Schmidhuber, “Long Short-Term Memory”](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)
- [Vaswani et al., “Attention Is All You Need”](https://arxiv.org/abs/1706.03762)
- [Caruana, “Multitask Learning”](https://doi.org/10.1023/A:1007379606734)
- [NumPy `argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
- [PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch Smooth L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html)
